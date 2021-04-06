import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path

from utils.utils import dict_update,thd_img,img_overlap
from utils.utils import pltImshow, saveImg, toNumpy
from utils.utils import precisionRecall_torch,get_model
from utils.utils import save_checkpoint,pretrainedLoader
from utils.utils import labels2Dto3D, flattenDetection, labels2Dto3D_flattened

class Train_model_frontend(object):
    default_config = {
        "train_iter": 170000,
        "save_interval": 2000,
        "tensorboard_interval": 200,
        "model": {"subpixel": {"enable": False}},
    }

    def __init__(self, config, save_path=Path("."), device="cpu", verbose=False):
        """
        ## default dimension:
            heatmap: torch (batch_size, H, W, 1)
            dense_desc: torch (batch_size, H, W, 256)
            pts: [batch_size, np (N, 3)]
            desc: [batch_size, np(256, N)]
        
        :param config:
            dense_loss, sparse_loss (default)
            
        """
        # config
        print("Load Train_model_frontend!!")
        self.config = self.default_config
        self.config = dict_update(self.config, config)
        print("check config!!", self.config)

        # init parameters
        self.device = device
        self.save_path = save_path
        self._train = True
        self._eval = True
        self.cell_size = 8
        self.subpixel = False
        self.loss = 0

        self.max_iter = config["train_iter"]

        if self.config["model"]["dense_loss"]["enable"]:
            ## original superpoint paper uses dense loss
            print("use dense_loss!")
            from utils.utils import descriptor_loss

            self.desc_params = self.config["model"]["dense_loss"]["params"]
            self.descriptor_loss = descriptor_loss
            self.desc_loss_type = "dense"
        elif self.config["model"]["sparse_loss"]["enable"]:
            ## our sparse loss has similar performace, more efficient
            print("use sparse_loss!")
            self.desc_params = self.config["model"]["sparse_loss"]["params"]
            from utils.loss_functions.sparse_loss import batch_descriptor_loss_sparse

            self.descriptor_loss = batch_descriptor_loss_sparse
            self.desc_loss_type = "sparse"

        if self.config["model"]["subpixel"]["enable"]:
            ## deprecated: only for testing subpixel prediction
            self.subpixel = True

            def get_func(path, name):
                logging.info("=> from %s import %s", path, name)
                mod = __import__("{}".format(path), fromlist=[""])
                return getattr(mod, name)

            self.subpixel_loss_func = get_func(
                "utils.losses", self.config["model"]["subpixel"]["loss_func"]
            )
        self.printImportantConfig()

        pass

    def train(self, **options):
        logging.info("n_iter: %d", self.n_iter)
        logging.info("max_iter: %d", self.max_iter)
        running_losses = []
        epoch = 0
        while self.n_iter < self.max_iter:
            print("epoch: ", epoch)
            epoch += 1
            for i, sample_train in tqdm(enumerate(self.train_loader)):
                loss_out = self.train_val_sample(sample_train, self.n_iter, True)
                self.n_iter += 1
                running_losses.append(loss_out)
                if self._eval and self.n_iter % self.config["validation_interval"] == 0:
                    logging.info("====== Validating...")
                    for j, sample_val in enumerate(self.val_loader):
                        self.train_val_sample(sample_val, self.n_iter + j, False)
                        if j > self.config.get("validation_size", 3):
                            break
                # save model
                if self.n_iter % self.config["save_interval"] == 0:
                    logging.info(
                        "save model: every %d interval, current iteration: %d",
                        self.config["save_interval"],
                        self.n_iter,
                    )
                    self.saveModel()
                if self.n_iter > self.max_iter:
                    logging.info("End training: %d", self.n_iter)
                    break
        pass

    def train_val_sample(self, sample, n_iter=0, train=False):
        task = "train" if train else "val"
        tb_interval = self.config["tensorboard_interval"]

        losses = {}
        img, labels_2D, mask_2D = (
            sample["image"],
            sample["labels_2D"],
            sample["valid_mask"],
        )
        batch_size, H, W = img.shape[0], img.shape[2], img.shape[3]
        self.batch_size = batch_size
        Hc = H // self.cell_size
        Wc = W // self.cell_size

        # warped images
        img_warp, labels_warp_2D, mask_warp_2D = (
            sample["warped_img"],
            sample["warped_labels"],
            sample["warped_valid_mask"],
        )

        mat_H, mat_H_inv = sample["homographies"], sample["inv_homographies"]

        self.optimizer.zero_grad()

        # forward + backward + optimize
        if train:
            outs, outs_warp = (
                self.net(img.to(self.device)),
                self.net(img_warp.to(self.device), subpixel=self.subpixel),
            )
            semi, coarse_desc = outs[0], outs[1]
            semi_warp, coarse_desc_warp = outs_warp[0], outs_warp[1]
        else:
            with torch.no_grad():
                outs, outs_warp = (
                    self.net(img.to(self.device)),
                    self.net(img_warp.to(self.device), subpixel=self.subpixel),
                )
                semi, coarse_desc = outs[0], outs[1]
                semi_warp, coarse_desc_warp = outs_warp[0], outs_warp[1]
                pass

        # detector loss
        labels3D_in_loss = self.getLabels(labels_2D, self.cell_size, device=self.device)
        mask_3D_flattened = self.getMasks(mask_2D, self.cell_size, device=self.device)
        loss_det = self.get_loss(
            semi, labels3D_in_loss, mask_3D_flattened, device=self.device
        )

        ## warping
        labels3D_in_loss = self.getLabels(
            labels_warp_2D, self.cell_size, device=self.device
        )
        mask_3D_flattened = self.getMasks(
            mask_warp_2D, self.cell_size, device=self.device
        )
        loss_det_warp = self.get_loss(
            semi_warp, labels3D_in_loss, mask_3D_flattened, device=self.device
        )

        mask_desc = mask_3D_flattened.unsqueeze(1)

        # descriptor loss
        loss_desc, mask, positive_dist, negative_dist = self.descriptor_loss(
            coarse_desc,
            coarse_desc_warp,
            mat_H,
            mask_valid=mask_desc,
            device=self.device,
            **self.desc_params
        )

        loss = (
            loss_det + loss_det_warp + self.config["model"]["lambda_loss"] * loss_desc
        )

        if self.subpixel:
            dense_map = flattenDetection(semi_warp)  # tensor [batch, 1, H, W]
            concat_features = torch.cat(
                (img_warp.to(self.device), dense_map), dim=1
            )  # tensor [batch, n, H, W]

            pred_heatmap = outs_warp[2]  # tensor [batch, 1, H, W]
            labels_warped_res = sample["warped_res"]

            subpix_loss = self.subpixel_loss_func(
                labels_warp_2D.to(self.device),
                labels_warped_res.to(self.device),
                pred_heatmap.to(self.device),
                patch_size=11,
            )

            label_idx = labels_2D[...].nonzero()
            from utils.losses import extract_patches

            patch_size = 32
            patches = extract_patches(
                label_idx.to(self.device),
                img_warp.to(self.device),
                patch_size=patch_size,
            )  
            print("patches: ", patches.shape)

            def label_to_points(labels_res, points):
                labels_res = labels_res.transpose(1, 2).transpose(2, 3).unsqueeze(1)
                points_res = labels_res[
                    points[:, 0], points[:, 1], points[:, 2], points[:, 3], :
                ]  # tensor [N, 2]
                return points_res

            points_res = label_to_points(labels_warped_res, label_idx)

            num_patches_max = 500
            pred_res = self.subnet(
                patches[:num_patches_max, ...].to(self.device)
            )  

            # loss function
            def get_loss(points_res, pred_res):
                loss = points_res - pred_res
                loss = torch.norm(loss, p=2, dim=-1).mean()
                return loss

            loss = get_loss(points_res[:num_patches_max, ...].to(self.device), pred_res)
            losses.update({"subpix_loss": subpix_loss})

        self.loss = loss
        losses.update(
            {
                "loss": loss,
                "loss_det": loss_det,
                "loss_det_warp": loss_det_warp,
                "loss_det": loss_det,
                "loss_det_warp": loss_det_warp,
                "positive_dist": positive_dist,
                "negative_dist": negative_dist,
            }
        )

        if train:
            loss.backward()
            self.optimizer.step()

        self.addLosses2tensorboard(losses, task)
        if n_iter % tb_interval == 0 or task == "val":
            logging.info(
                "current iteration: %d, tensorboard_interval: %d", n_iter, tb_interval
            )
            self.addImg2tensorboard(
                img,
                labels_2D,
                semi,
                img_warp,
                labels_warp_2D,
                mask_warp_2D,
                semi_warp,
                mask_3D_flattened=mask_3D_flattened,
                task=task,
            )

            if self.subpixel:
                self.add_single_image_to_tb(
                    task, pred_heatmap, n_iter, name="subpixel_heatmap"
                )

            self.printLosses(losses, task)
            self.add2tensorboard_nms(
                img, labels_2D, semi, task=task, batch_size=batch_size
            )

        return loss.item()

    def dataParallel(self):
        print("=== Let's use", torch.cuda.device_count(), "GPUs!")
        self.net = nn.DataParallel(self.net)
        self.optimizer = self.adamOptim(
            self.net, lr=self.config["model"]["learning_rate"]
        )
        pass

    def adamOptim(self, net, lr):
        print("adam optimizer")
        import torch.optim as optim
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
        return optimizer

    def loadModel(self):
        model = self.config["model"]["name"]
        descriptor_length = self.config["model"]["descriptor_length"]
        print("model: ", model)
        net = get_model(model)
        net = net(descriptor_length).to(self.device)
        logging.info("=> setting adam solver")
        optimizer = self.adamOptim(net, lr=self.config["model"]["learning_rate"])

        n_iter = 0
        if self.config["retrain"] == True:
            logging.info("New model")
            pass
        else:
            path = self.config["pretrained"]
            mode = "" if path[-4:] == ".pth" else "full" # the suffix is '.pth' or 'tar.gz'
            logging.info("load pretrained model from: %s", path)
            net, optimizer, n_iter = pretrainedLoader(
                net, optimizer, n_iter, path, mode=mode, full_path=True
            )
            logging.info("successfully load pretrained model from: %s", path)

        def setIter(n_iter):
            if self.config["reset_iter"]:
                logging.info("reset iterations to 0")
                n_iter = 0
            return n_iter

        self.net = net
        self.optimizer = optimizer
        self.n_iter = setIter(n_iter)
        pass

    def saveModel(self):
        model_state_dict = self.net.state_dict()#单卡保存
        # model_state_dict = self.net.module.state_dict()#多卡保存
        save_checkpoint(
            self.save_path,
            {
                "n_iter": self.n_iter + 1,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.loss,
            },
            self.n_iter,
        )
        pass

    def getLabels(self, labels_2D, cell_size, device="cpu"):
        labels3D_flattened = labels2Dto3D_flattened(
            labels_2D.to(device), cell_size=cell_size
        )
        labels3D_in_loss = labels3D_flattened
        return labels3D_in_loss

    def getMasks(self, mask_2D, cell_size, device="cpu"):
        """
        # 2D mask is constructed into 3D (Hc, Wc) space for training
        :param mask_2D:
            tensor [batch, 1, H, W]
        :param cell_size:
            8 (default)
        :param device:
        :return:
            flattened 3D mask for training
        """
        mask_3D = labels2Dto3D(
            mask_2D.to(device), cell_size=cell_size, add_dustbin=False
        ).float()#(batch,64,15,20)
        mask_3D_flattened = torch.prod(mask_3D, 1)#(batch,15,20),why product?
        return mask_3D_flattened

    def get_loss(self, semi, labels3D_in_loss, mask_3D_flattened, device="cpu"):
        loss_func = nn.CrossEntropyLoss(reduce=False).to(device)
        loss = loss_func(semi, labels3D_in_loss)
        loss = (loss * mask_3D_flattened).sum()
        loss = loss / (mask_3D_flattened.sum() + 1e-10)
        return loss

    def get_heatmap(self, semi, det_loss_type="softmax"):
        if det_loss_type == "l2":
            heatmap = self.flatten_64to1(semi)
        else:
            heatmap = flattenDetection(semi)
        return heatmap

    def printLosses(self, losses, task="training"):
        for element in list(losses):
            print(task, "-", element, ": ", losses[element].item())

    def printImportantConfig(self):
        """
        # print important configs
        :return:
        """
        print("=" * 10, " check!!! ", "=" * 10)
        print("learning_rate: ", self.config["model"]["learning_rate"])
        print("lambda_loss: ", self.config["model"]["lambda_loss"])
        print("detection_threshold: ", self.config["model"]["detection_threshold"])
        print("batch_size: ", self.config["model"]["batch_size"])

        print("=" * 10, " descriptor: ", self.desc_loss_type, "=" * 10)
        for item in list(self.desc_params):
            print(item, ": ", self.desc_params[item])

        print("=" * 32)
        pass
    
    def tb_scalar_dict(self, losses, task="training"):
        for element in list(losses):
            self.writer.add_scalar(task + "-" + element, losses[element], self.n_iter)

    def tb_images_dict(self, task, tb_imgs, max_img=5):
        for element in list(tb_imgs):
            for idx in range(tb_imgs[element].shape[0]):
                if idx >= max_img:
                    break
                self.writer.add_image(
                    task + "-" + element + "/%d" % idx,
                    tb_imgs[element][idx, ...],
                    self.n_iter,
                )

    def tb_hist_dict(self, task, tb_dict):
        for element in list(tb_dict):
            self.writer.add_histogram(
                task + "-" + element, tb_dict[element], self.n_iter
            )
        pass

    def add_single_image_to_tb(self, task, img_tensor, n_iter, name="img"):
        if img_tensor.dim() == 4:
            for i in range(min(img_tensor.shape[0], 5)):
                self.writer.add_image(
                    task + "-" + name + "/%d" % i, img_tensor[i, :, :, :], n_iter
                )
        else:
            self.writer.add_image(task + "-" + name, img_tensor[:, :, :], n_iter)

    def addImg2tensorboard(
        self,
        img,
        labels_2D,
        semi,
        img_warp=None,
        labels_warp_2D=None,
        mask_warp_2D=None,
        semi_warp=None,
        mask_3D_flattened=None,
        task="training"):
        n_iter = self.n_iter
        semi_flat = flattenDetection(semi[0, :, :, :])
        semi_warp_flat = flattenDetection(semi_warp[0, :, :, :])

        thd = self.config["model"]["detection_threshold"]
        semi_thd = thd_img(semi_flat, thd=thd)
        semi_warp_thd = thd_img(semi_warp_flat, thd=thd)

        result_overlap = img_overlap(
            toNumpy(labels_2D[0, :, :, :]), toNumpy(semi_thd), toNumpy(img[0, :, :, :])
        )

        self.writer.add_image(
            task + "-detector_output_thd_overlay", result_overlap, n_iter
        )
        saveImg(
            result_overlap.transpose([1, 2, 0])[..., [2, 1, 0]] * 255, "test_0.png"
        )  # rgb to bgr * 255

        result_overlap = img_overlap(
            toNumpy(labels_warp_2D[0, :, :, :]),
            toNumpy(semi_warp_thd),
            toNumpy(img_warp[0, :, :, :]),
        )
        self.writer.add_image(
            task + "-warp_detector_output_thd_overlay", result_overlap, n_iter
        )
        saveImg(
            result_overlap.transpose([1, 2, 0])[..., [2, 1, 0]] * 255, "test_1.png"
        )  # rgb to bgr * 255

        mask_overlap = img_overlap(
            toNumpy(1 - mask_warp_2D[0, :, :, :]) / 2,
            np.zeros_like(toNumpy(img_warp[0, :, :, :])),
            toNumpy(img_warp[0, :, :, :]),
        )

        for i in range(self.batch_size):
            if i < 5:
                self.writer.add_image(
                    task + "-mask_warp_origin", mask_warp_2D[i, :, :, :], n_iter
                )
                self.writer.add_image(
                    task + "-mask_warp_3D_flattened", mask_3D_flattened[i, :, :], n_iter
                )
        self.writer.add_image(task + "-mask_warp_overlay", mask_overlap, n_iter)

    def add2tensorboard_nms(self, img, labels_2D, semi, task="training", batch_size=1):
        from utils.utils import getPtsFromHeatmap
        from utils.utils import box_nms

        boxNms = False
        n_iter = self.n_iter
        nms_dist = self.config["model"]["nms"]
        conf_thresh = self.config["model"]["detection_threshold"]
        precision_recall_list = []
        precision_recall_boxnms_list = []
        for idx in range(batch_size):
            semi_flat_tensor = flattenDetection(semi[idx, :, :, :]).detach()
            semi_flat = toNumpy(semi_flat_tensor)
            semi_thd = np.squeeze(semi_flat, 0)
            pts_nms = getPtsFromHeatmap(semi_thd, conf_thresh, nms_dist)
            semi_thd_nms_sample = np.zeros_like(semi_thd)
            semi_thd_nms_sample[
                pts_nms[1, :].astype(np.int), pts_nms[0, :].astype(np.int)
            ] = 1

            label_sample = torch.squeeze(labels_2D[idx, :, :, :])
            label_sample_nms_sample = label_sample

            if idx < 5:
                result_overlap = img_overlap(
                    np.expand_dims(label_sample_nms_sample, 0),
                    np.expand_dims(semi_thd_nms_sample, 0),
                    toNumpy(img[idx, :, :, :]),
                )
                self.writer.add_image(
                    task + "-detector_output_thd_overlay-NMS" + "/%d" % idx,
                    result_overlap,
                    n_iter,
                )
            assert semi_thd_nms_sample.shape == label_sample_nms_sample.size()
            precision_recall = precisionRecall_torch(
                torch.from_numpy(semi_thd_nms_sample), label_sample_nms_sample
            )
            precision_recall_list.append(precision_recall)

            if boxNms:
                semi_flat_tensor_nms = box_nms(
                    semi_flat_tensor.squeeze(), nms_dist, min_prob=conf_thresh
                ).cpu()
                semi_flat_tensor_nms = (semi_flat_tensor_nms >= conf_thresh).float()

                if idx < 5:
                    result_overlap = img_overlap(
                        np.expand_dims(label_sample_nms_sample, 0),
                        semi_flat_tensor_nms.numpy()[np.newaxis, :, :],
                        toNumpy(img[idx, :, :, :]),
                    )
                    self.writer.add_image(
                        task + "-detector_output_thd_overlay-boxNMS" + "/%d" % idx,
                        result_overlap,
                        n_iter,
                    )
                precision_recall_boxnms = precisionRecall_torch(
                    semi_flat_tensor_nms, label_sample_nms_sample
                )
                precision_recall_boxnms_list.append(precision_recall_boxnms)

        precision = np.mean(
            [
                precision_recall["precision"]
                for precision_recall in precision_recall_list
            ]
        )
        recall = np.mean(
            [precision_recall["recall"] for precision_recall in precision_recall_list]
        )
        self.writer.add_scalar(task + "-precision_nms", precision, n_iter)
        self.writer.add_scalar(task + "-recall_nms", recall, n_iter)
        print(
            "-- [%s-%d-fast NMS] precision: %.4f, recall: %.4f"
            % (task, n_iter, precision, recall)
        )
        if boxNms:
            precision = np.mean(
                [
                    precision_recall["precision"]
                    for precision_recall in precision_recall_boxnms_list
                ]
            )
            recall = np.mean(
                [
                    precision_recall["recall"]
                    for precision_recall in precision_recall_boxnms_list
                ]
            )
            self.writer.add_scalar(task + "-precision_boxnms", precision, n_iter)
            self.writer.add_scalar(task + "-recall_boxnms", recall, n_iter)
            print(
                "-- [%s-%d-boxNMS] precision: %.4f, recall: %.4f"
                % (task, n_iter, precision, recall)
            )

    @property
    def writer(self):
        return self._writer

    @writer.setter
    def writer(self, writer):
        print("set writer")
        self._writer = writer

    @property
    def train_loader(self):
        print("get dataloader")
        return self._train_loader

    @train_loader.setter
    def train_loader(self, loader):
        print("set train loader")
        self._train_loader = loader

    @property
    def val_loader(self):
        print("get dataloader")
        return self._val_loader

    @val_loader.setter
    def val_loader(self, loader):
        print("set train loader")
        self._val_loader = loader

    ######## static methods ########
    @staticmethod
    def input_to_imgDict(sample, tb_images_dict):
        for e in list(sample):
            element = sample[e]
            if type(element) is torch.Tensor:
                if element.dim() == 4:
                    tb_images_dict[e] = element
        return tb_images_dict

    @staticmethod
    def interpolate_to_dense(coarse_desc, cell_size=8):
        dense_desc = nn.functional.interpolate(
            coarse_desc, scale_factor=(cell_size, cell_size), mode="bilinear"
        )
        def norm_desc(desc):
            dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
            desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
            return desc
        dense_desc = norm_desc(dense_desc)
        return dense_desc
