import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        mid_channels = mid_channels or out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 6, out_classes: int = 5):
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = out_classes

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = DecoderBlock(512, 512, 256)
        self.up2 = DecoderBlock(256, 256, 128)
        self.up3 = DecoderBlock(128, 128, 64)
        self.up4 = DecoderBlock(64, 64, 64)
        self.outc = OutConv(64, out_classes)

    def get_gradcam_target_layer(self) -> nn.Module:
        return self.up4.conv.conv2

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        d1 = self.up1(x5, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)
        logits = self.outc(d4)

        if return_features:
            return logits, {
                "encoder": {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5},
                "decoder": {"d1": d1, "d2": d2, "d3": d3, "d4": d4},
            }
        return logits


class SiameseUNet(nn.Module):
    """
    Shared-encoder change-detection U-Net that combines post-disaster features with
    absolute pre/post differences. Optional tabular inputs modulate the bottleneck.
    """

    def __init__(self, in_channels: int = 3, out_classes: int = 5, tabular_dim: int = 0):
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = out_classes
        self.tabular_dim = tabular_dim

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        # Each fusion stage concatenates the post-disaster feature map with |post - pre|.
        self.up1 = DecoderBlock(1024, 1024, 512)
        self.up2 = DecoderBlock(512, 512, 256)
        self.up3 = DecoderBlock(256, 256, 128)
        self.up4 = DecoderBlock(128, 128, 64)
        self.outc = OutConv(64, out_classes)

        if tabular_dim > 0:
            self.tabular_film = nn.Sequential(
                nn.Linear(tabular_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2048),
            )
        else:
            self.tabular_film = None

    def encode(self, x: torch.Tensor):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5

    @staticmethod
    def _fuse_features(pre_features, post_features):
        fused = []
        for pre_feature, post_feature in zip(pre_features, post_features):
            fused.append(torch.cat([post_feature, torch.abs(post_feature - pre_feature)], dim=1))
        return fused

    def _apply_tabular_conditioning(
        self, features: torch.Tensor, tabular: torch.Tensor = None
    ) -> torch.Tensor:
        if self.tabular_film is None or tabular is None:
            return features

        gamma_beta = self.tabular_film(tabular)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return features * (1.0 + gamma) + beta

    def get_gradcam_target_layer(self) -> nn.Module:
        return self.up4.conv.conv2

    def forward(
        self,
        x: torch.Tensor,
        tabular: torch.Tensor = None,
        return_features: bool = False,
    ):
        pre_image = x[:, :3, :, :]
        post_image = x[:, 3:, :, :]

        pre_features = self.encode(pre_image)
        post_features = self.encode(post_image)
        fused_features = self._fuse_features(pre_features, post_features)

        bottleneck = self._apply_tabular_conditioning(fused_features[-1], tabular)

        d1 = self.up1(bottleneck, fused_features[-2])
        d2 = self.up2(d1, fused_features[-3])
        d3 = self.up3(d2, fused_features[-4])
        d4 = self.up4(d3, fused_features[-5])
        logits = self.outc(d4)

        if return_features:
            return logits, {
                "pre_encoder": {
                    "x1": pre_features[0],
                    "x2": pre_features[1],
                    "x3": pre_features[2],
                    "x4": pre_features[3],
                    "x5": pre_features[4],
                },
                "post_encoder": {
                    "x1": post_features[0],
                    "x2": post_features[1],
                    "x3": post_features[2],
                    "x4": post_features[3],
                    "x5": post_features[4],
                },
                "fused": {
                    "x1": fused_features[0],
                    "x2": fused_features[1],
                    "x3": fused_features[2],
                    "x4": fused_features[3],
                    "x5": fused_features[4],
                },
                "decoder": {"d1": d1, "d2": d2, "d3": d3, "d4": d4},
                "bottleneck": bottleneck,
            }
        return logits


class MultimodalDamageAssessment(nn.Module):
    """
    Convenience wrapper that exposes a multimodal Siamese U-Net using tabular weather
    and terrain inputs during segmentation.
    """

    def __init__(self, vision_model: nn.Module = None, tabular_features: int = 4, out_classes: int = 5):
        super().__init__()
        self.vision_model = vision_model or SiameseUNet(
            in_channels=3,
            out_classes=out_classes,
            tabular_dim=tabular_features,
        )

    def get_gradcam_target_layer(self) -> nn.Module:
        if hasattr(self.vision_model, "get_gradcam_target_layer"):
            return self.vision_model.get_gradcam_target_layer()
        raise AttributeError("Wrapped vision model does not expose a Grad-CAM target layer.")

    def forward(self, img: torch.Tensor, tabular: torch.Tensor, return_features: bool = False):
        return self.vision_model(img, tabular, return_features=return_features)


class RiskLSTM(nn.Module):
    def __init__(self, input_size: int = 4, hidden_size: int = 64, num_layers: int = 2, out_size: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)


class SimpleViTSegmentation(nn.Module):
    def __init__(self, in_channels: int = 6, out_classes: int = 5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),
        )
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(64, out_classes, kernel_size=1),
        )

    def get_gradcam_target_layer(self) -> nn.Module:
        return self.decoder[0]

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self.stem(x)
        batch_size, channels, height, width = x.shape
        flattened = x.flatten(2).transpose(1, 2)
        attended, _ = self.attention(flattened, flattened, flattened)
        attended = attended.transpose(1, 2).reshape(batch_size, channels, height, width)
        logits = self.decoder(attended)
        if return_features:
            return logits, {"stem": x, "attention": attended}
        return logits
