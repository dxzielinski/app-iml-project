import math
import torch
import torchmetrics
import lightning as L
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from model.config import MasterConfig


DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(
            num_output_channels=MasterConfig.IN_CHANNELS
        ),  # ensure 1 channel
        transforms.Resize((MasterConfig.IMG_WIDTH, MasterConfig.IMG_HEIGHT)),
        transforms.ToTensor(),
    ]
)


def _calc_conv_output_size(size, kernel_size, stride, padding, pooling_size):
    conv_out = math.floor((size - kernel_size + 2 * padding) / stride + 1)
    pool_out = math.floor(conv_out / pooling_size)
    if pool_out <= 1:
        raise ValueError("Dimensions are shrinking too much after pooling")
    return pool_out


def custom_init_weights_clojure(hyperparameter_space: dict):
    """
    Wrapper function to pass hyperparameter_space to init_weights.
    """

    def init_weights(m):
        """
        Apply custom weight initialization to the model.
        """
        if isinstance(m, torch.nn.Conv2d):
            weight_init_conv = hyperparameter_space["weight_init_conv"]
            if weight_init_conv == "kaiming_normal":
                torch.nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity=hyperparameter_space["activation_fn"],
                )
            if weight_init_conv == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity=hyperparameter_space["activation_fn"],
                )
            if weight_init_conv == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            if weight_init_conv == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            if weight_init_conv == "uniform":
                torch.nn.init.uniform_(m.weight)
            if weight_init_conv == "normal":
                torch.nn.init.normal_(m.weight)
        if isinstance(m, torch.nn.Linear):
            weight_init_fc = hyperparameter_space["weight_init_fc"]
            if weight_init_fc == "kaiming_normal":
                torch.nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity=hyperparameter_space["activation_fn"],
                )
            if weight_init_fc == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity=hyperparameter_space["activation_fn"],
                )
            if weight_init_fc == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            if weight_init_fc == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            if weight_init_fc == "uniform":
                torch.nn.init.uniform_(m.weight)
            if weight_init_fc == "normal":
                torch.nn.init.normal_(m.weight)

    return init_weights


class CustomCNN(torch.nn.Module):
    def __init__(self, hyperparameter_space: dict):
        super().__init__()
        layers: list[torch.nn.Module] = []
        dropout_rates: list[float] = hyperparameter_space["dropout_rates"]
        conv_layers_dims: list[int] = hyperparameter_space["conv_layers_dims"]
        fc_layers_dims: list[int] = hyperparameter_space["fc_layers_dims"]
        batch_norm: bool = hyperparameter_space["batch_norm"]
        kernel_size: int = hyperparameter_space["kernel_size"]
        stride: int = hyperparameter_space["stride"]
        padding: int = hyperparameter_space["padding"]
        pooling_size: int = hyperparameter_space["pooling_size"]
        if hyperparameter_space["activation_fn"] == "relu":
            activation_fn = torch.nn.ReLU()
        if hyperparameter_space["activation_fn"] == "leaky_relu":
            activation_fn = torch.nn.LeakyReLU()
        if hyperparameter_space["activation_fn"] == "elu":
            activation_fn = torch.nn.ELU()
        if hyperparameter_space["activation_fn"] == "gelu":
            activation_fn = torch.nn.GELU()
        if hyperparameter_space["activation_fn"] == "sigmoid":
            activation_fn = torch.nn.Sigmoid()
        in_channels = MasterConfig.IN_CHANNELS
        width = MasterConfig.IMG_WIDTH
        height = MasterConfig.IMG_HEIGHT
        for conv_layer_dim in conv_layers_dims:
            layers.append(
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=conv_layer_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                )
            )
            if batch_norm is True:
                layers.append(torch.nn.BatchNorm2d(conv_layer_dim))
            layers.append(activation_fn)
            layers.append(
                torch.nn.MaxPool2d(kernel_size=pooling_size, stride=pooling_size)
            )
            width = _calc_conv_output_size(
                width, kernel_size, stride, padding, pooling_size
            )
            height = _calc_conv_output_size(
                height, kernel_size, stride, padding, pooling_size
            )
            in_channels = conv_layer_dim

        layers.append(torch.nn.Flatten())
        mult_factor = width * height
        i = 0
        for fc_layer_dim in fc_layers_dims:
            layers.append(
                torch.nn.Linear(
                    in_features=in_channels * mult_factor, out_features=fc_layer_dim
                )
            )
            if batch_norm is True:
                layers.append(torch.nn.BatchNorm1d(fc_layer_dim))
            layers.append(activation_fn)
            layers.append(torch.nn.Dropout(dropout_rates[i]))
            mult_factor = 1
            in_channels = fc_layer_dim
            i += 1
        layers.append(
            torch.nn.Linear(
                in_features=in_channels, out_features=MasterConfig.NUM_CLASSES
            )
        )
        self.layers = torch.nn.Sequential(*layers)
        self.apply(custom_init_weights_clojure(hyperparameter_space))

    def forward(self, x):
        return self.layers(x)


class Classifier(L.LightningModule):

    def __init__(
        self,
        hyperparameter_space: dict,
        learning_rate: float,
        num_classes=MasterConfig.NUM_CLASSES,
        task=MasterConfig.TASK,
    ):
        super().__init__()
        self.model = CustomCNN(hyperparameter_space=hyperparameter_space)
        self.learning_rate = learning_rate
        if hyperparameter_space["loss_fn"] == "CrossEntropyLoss":
            self.loss_fn = torch.nn.CrossEntropyLoss()
        if hyperparameter_space["optimizer"] == "Adam":
            self.optimizer = torch.optim.Adam
        if hyperparameter_space["optimizer"] == "AdamW":
            self.optimizer = torch.optim.AdamW
        if hyperparameter_space["optimizer"] == "SGD":
            self.optimizer = torch.optim.SGD
        if hyperparameter_space["optimizer"] == "RMSprop":
            self.optimizer = torch.optim.RMSprop
        self.train_metrics = torchmetrics.MetricCollection(
            {
                "f1_macro": torchmetrics.F1Score(
                    task=task, num_classes=num_classes, average="macro"
                ),
                "precision": torchmetrics.Precision(
                    task=task, num_classes=num_classes, average="macro"
                ),
                "recall": torchmetrics.Recall(
                    task=task, num_classes=num_classes, average="macro"
                ),
                "auroc": torchmetrics.AUROC(
                    task=task, num_classes=num_classes, average="macro"
                ),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")
        self.train_batch_outputs = []
        self.val_batch_outputs = []
        self.test_batch_outputs = []
        self.custom_hparams = hyperparameter_space

    def on_train_start(self):
        if self.logger is not None:
            self.logger.log_hyperparams(self.custom_hparams)

    def on_test_start(self):
        if self.logger is not None:
            self.logger.log_hyperparams(self.custom_hparams)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probabilities = torch.softmax(logits, dim=1)
        self.train_batch_outputs.append({"probabilities": probabilities, "y": y})
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        probabilities = torch.cat(
            [x["probabilities"] for x in self.train_batch_outputs]
        )
        y = torch.cat([x["y"] for x in self.train_batch_outputs])
        metrics = self.train_metrics(probabilities, y)
        self.log_dict(metrics)
        self.train_metrics.reset()
        self.train_batch_outputs.clear()

    def validation_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probabilities = torch.softmax(logits, dim=1)
        self.val_batch_outputs.append({"probabilities": probabilities, "y": y})
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        probabilities = torch.cat([x["probabilities"] for x in self.val_batch_outputs])
        y = torch.cat([x["y"] for x in self.val_batch_outputs])
        metrics = self.val_metrics(probabilities, y)
        self.log_dict(metrics)
        self.val_metrics.reset()
        self.val_batch_outputs.clear()

    def test_step(self, batch):
        x, y = batch
        logits = self(x)
        probabilities = torch.softmax(logits, dim=1)
        self.test_batch_outputs.append({"probabilities": probabilities, "y": y})

    def on_test_epoch_end(self):
        probabilities = torch.cat([x["probabilities"] for x in self.test_batch_outputs])
        y = torch.cat([x["y"] for x in self.test_batch_outputs])
        metrics = self.test_metrics(probabilities, y)
        self.log_dict(metrics)
        self.test_metrics.reset()
        self.test_batch_outputs.clear()

    def configure_optimizers(self):
        return self.optimizer(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.custom_hparams["weight_decay"],
        )


class ClassificationData(L.LightningDataModule):

    def __init__(self, data_dir=MasterConfig.DATA_DIR, batch_size=16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = ImageFolder(
                root=f"{self.data_dir}/train", transform=DEFAULT_TRANSFORM
            )
            self.val_dataset = ImageFolder(
                root=f"{self.data_dir}/val", transform=DEFAULT_TRANSFORM
            )
        if stage == "test":
            self.test_dataset = ImageFolder(
                root=f"{self.data_dir}/test", transform=DEFAULT_TRANSFORM
            )
        if stage == "predict":
            self.predict_dataset = ImageFolder(
                root=f"{self.data_dir}/predict", transform=DEFAULT_TRANSFORM
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=16,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=16,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=16,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=16,
            persistent_workers=True,
        )
