import torch

from src.dataset import DisasterAssessmentDataset, generate_mock_data, get_val_transforms
from src.evaluate import evaluate_model
from src.model import MultimodalDamageAssessment, RiskLSTM, SiameseUNet, SimpleViTSegmentation, UNet
from src.simulate import simulate_real_time_stream
from src.train import DiceCrossEntropyLoss, validate
from src.xai import generate_gradcam, select_gradcam_layer


def run_tests():
    print("Test 1: Synthetic smoke dataset...")
    metadata = generate_mock_data("debug_data", num_samples=2)
    dataset = DisasterAssessmentDataset(metadata, "debug_data", transform=get_val_transforms())
    images, tabular, masks = dataset[0]
    images = images.unsqueeze(0)
    tabular = tabular.unsqueeze(0)
    masks = masks.unsqueeze(0)
    print("Dataset loaded successfully.")

    print("Test 2: U-Net forward...")
    unet = UNet(in_channels=6, out_classes=5).eval()
    with torch.no_grad():
        unet_out = unet(images)
    print(f"U-Net output shape: {tuple(unet_out.shape)}")

    print("Test 3: Siamese U-Net forward...")
    siamese = SiameseUNet(in_channels=3, out_classes=5).eval()
    with torch.no_grad():
        siamese_out = siamese(images)
    print(f"Siamese output shape: {tuple(siamese_out.shape)}")

    print("Test 4: Multimodal Siamese forward...")
    multimodal = MultimodalDamageAssessment(tabular_features=4, out_classes=5).eval()
    with torch.no_grad():
        multimodal_out = multimodal(images, tabular)
    print(f"Multimodal output shape: {tuple(multimodal_out.shape)}")

    print("Test 5: Vision transformer forward...")
    vit = SimpleViTSegmentation(in_channels=6, out_classes=5).eval()
    with torch.no_grad():
        vit_out = vit(images)
    print(f"ViT output shape: {tuple(vit_out.shape)}")

    print("Test 6: Risk LSTM...")
    lstm = RiskLSTM(input_size=4).eval()
    with torch.no_grad():
        lstm_out = lstm(tabular.unsqueeze(1))
    print(f"Risk output shape: {tuple(lstm_out.shape)}")

    print("Test 7: Loss and validation...")
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    criterion = DiceCrossEntropyLoss(num_classes=5)
    validation_metrics = validate(siamese, loader, criterion, torch.device("cpu"))
    print(f"Validation metrics: {validation_metrics}")

    print("Test 8: Benchmark helper...")
    benchmark_metrics = evaluate_model(unet, loader, torch.device("cpu"))
    print(f"Evaluation metrics: {benchmark_metrics}")

    print("Test 9: Grad-CAM...")
    cam = generate_gradcam(
        siamese,
        select_gradcam_layer(siamese),
        images.clone(),
    )
    print(f"Grad-CAM shape: {cam.shape}")

    print("Test 10: Stream simulator...")
    simulate_real_time_stream(metadata.head(1), "debug_data", delay_seconds=0.0)

    print("ALL TESTS PASSED WITH NO ERRORS.")


if __name__ == "__main__":
    run_tests()
