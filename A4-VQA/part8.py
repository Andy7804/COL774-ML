# part8.py - Initial training of VQA model with frozen ResNet
import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from modules.vqa_model import VQAModel
from utils.dataset import CLEVRA4Dataset
from utils.metrics import calculate_metrics


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(images, input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct_predictions += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(dataloader)
    accuracy = correct_predictions / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    metrics = calculate_metrics(all_labels, all_preds)
    return avg_loss, metrics


def save_checkpoint(model, optimizer, epoch, val_metrics, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_metrics': val_metrics
    }, path)
    print(f"Checkpoint saved at {path}")


def plot_metrics(train_losses, val_losses, train_accs, val_accs, save_dir):
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_plot.png'))
    print(f"Metrics plot saved to {save_dir}/metrics_plot.png")

import csv
def save_metrics_to_csv(train_losses, train_accs, val_losses, val_accs, path):
    file_exists = os.path.exists(path)
    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])  # Write header
        for epoch in range(len(train_losses)):
            writer.writerow([epoch + 1, train_losses[epoch], train_accs[epoch], val_losses[epoch], val_accs[epoch]])


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.save_path, exist_ok=True)

    plots_dir = os.path.join(args.save_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Dataset paths
    train_img_dir = os.path.join(args.dataset, "images", "trainA")
    train_q_file = os.path.join(args.dataset, "questions", "CLEVR_trainA_questions.json")
    val_img_dir = os.path.join(args.dataset, "images", "valA")
    val_q_file = os.path.join(args.dataset, "questions", "CLEVR_valA_questions.json")

    # Create datasets
    train_dataset = CLEVRA4Dataset(
        image_dir=train_img_dir,
        question_json=train_q_file,
        mode='train'
    )

    val_dataset = CLEVRA4Dataset(
        image_dir=val_img_dir,
        question_json=val_q_file,
        mode='val'
    )

    # Create dataloaders
    batch_size = 32
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True,
        prefetch_factor=4
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Get vocabulary size and number of classes
    vocab_size = len(train_dataset.tokenizer)
    num_classes = len(train_dataset.get_answer_vocab())
    print(f"Vocab size: {vocab_size}, Num classes: {num_classes}")

    # Initialize model
    model = VQAModel(vocab_size=vocab_size, num_classes=num_classes)

    # Ensure ResNet is frozen
    for param in model.image_encoder.resnet_backbone.parameters():
        param.requires_grad = False

    model.to(device)

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    criterion = nn.CrossEntropyLoss()

    # For tracking metrics
    num_epochs = 20
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_f1 = 0.0

    best_val_loss = float('inf')
    early_stopping_patience = 5
    loss_improvement_threshold = 0.001  # delta
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Evaluate
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_metrics['accuracy'])

        # Update learning rate
        scheduler.step(val_loss)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f}")
        save_metrics_to_csv(train_losses, train_accs, val_losses, val_accs, os.path.join(args.save_path, 'metrics.csv'))

        # Save best model
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            save_checkpoint(model, optimizer, epoch, val_metrics, os.path.join(args.save_path, "best_model.pth"))

         # Save model if val loss improves significantly
        if val_loss < best_val_loss - loss_improvement_threshold:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(model, optimizer, epoch, val_metrics, os.path.join(args.save_path, "best_loss_model.pth"))
        else:
            epochs_without_improvement += 1
            print(f"No significant improvement in val loss for {epochs_without_improvement} epoch(s).")

        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Plot and save metrics
    plot_metrics(train_losses, val_losses, train_accs, val_accs, plots_dir)
    print("Training completed. Best validation F1 score:", best_val_f1)


def load_metrics_from_csv(path):
    epochs = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    if os.path.exists(path):
        with open(path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                epochs.append(int(row[0]))
                train_losses.append(float(row[1]))
                train_accs.append(float(row[2]))
                val_losses.append(float(row[3]))
                val_accs.append(float(row[4]))
    return epochs, train_losses, train_accs, val_losses, val_accs


def train2(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.save_path, exist_ok=True)
    plots_dir = os.path.join(args.save_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Dataset paths
    train_img_dir = os.path.join(args.dataset, "images", "trainA")
    train_q_file = os.path.join(args.dataset, "questions", "CLEVR_trainA_questions.json")
    val_img_dir = os.path.join(args.dataset, "images", "valA")
    val_q_file = os.path.join(args.dataset, "questions", "CLEVR_valA_questions.json")

    train_dataset = CLEVRA4Dataset(train_img_dir, train_q_file, mode='train')
    val_dataset = CLEVRA4Dataset(val_img_dir, val_q_file, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    vocab_size = len(train_dataset.tokenizer)
    num_classes = len(train_dataset.get_answer_vocab())
    print(f"Vocab size: {vocab_size}, Num classes: {num_classes}")

    model = VQAModel(vocab_size=vocab_size, num_classes=num_classes)
    for param in model.image_encoder.resnet_backbone.parameters():
        param.requires_grad = False
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    criterion = nn.CrossEntropyLoss()

    # Load checkpoint
    checkpoint_path = os.path.join(args.model_path, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_f1 = checkpoint['val_metrics']['f1_score']
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Resume training
    num_epochs = 20
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    # epochs, train_losses, train_accs, val_losses, val_accs = load_metrics_from_csv(os.path.join(args.save_path, 'metrics.csv'))

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_metrics['accuracy'])

        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f}")
        save_metrics_to_csv(train_losses, train_accs, val_losses, val_accs, os.path.join(args.save_path, 'metrics.csv'))


        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            save_checkpoint(model, optimizer, epoch, val_metrics, os.path.join(args.save_path, "best_model.pth"))

        if val_loss < best_val_loss - 0.001:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(model, optimizer, epoch, val_metrics, os.path.join(args.save_path, "best_loss_model.pth"))
        else:
            epochs_without_improvement += 1
            print(f"No significant improvement in val loss for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= 5:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    plot_metrics(train_losses, val_losses, train_accs, val_accs, plots_dir)
    print("Resumed training completed. Best validation F1 score:", best_val_f1)


def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test dataset
    test_img_dir = os.path.join(args.dataset, "images", "testA")
    test_q_file = os.path.join(args.dataset, "questions", "CLEVR_testA_questions.json")

    test_dataset = CLEVRA4Dataset(
        image_dir=test_img_dir,
        question_json=test_q_file,
        mode='test'
    )

    # Create dataloader
    batch_size = 32
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Get vocabulary size and number of classes
    vocab_size = len(test_dataset.tokenizer)
    num_classes = len(test_dataset.get_answer_vocab())

    # Initialize model and load weights
    model = VQAModel(vocab_size=vocab_size, num_classes=num_classes)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Evaluate on test set
    criterion = nn.CrossEntropyLoss()
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)

    # Print metrics
    print("\nTest Set Metrics:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1_score']:.4f}")

    # Visualize some predictions
    idx_to_ans = {idx: ans for ans, idx in test_dataset.answer2idx.items()}

    # Function to visualize predictions
    def visualize_predictions(dataloader, model, device, idx_to_ans, num_samples=5, correct=True):
        model.eval()
        all_samples = []
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(images, input_ids, attention_mask)
                preds = outputs.argmax(dim=1)

                # Find examples that match our criteria (correct or incorrect predictions)
                for i in range(len(preds)):
                    if (preds[i] == labels[i]) == correct:
                        all_samples.append({
                            'image': images[i].cpu(),
                            'question': test_dataset.tokenizer.decode(input_ids[i], skip_special_tokens=True),
                            'pred': idx_to_ans[preds[i].item()],
                            'label': idx_to_ans[labels[i].item()]
                        })
                        if len(all_samples) >= num_samples:
                            break

                if len(all_samples) >= num_samples:
                    break

        # Visualize
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 5*num_samples))
        if num_samples == 1:
            axes = [axes]

        for i, sample in enumerate(all_samples):
            # Denormalize image
            img = sample['image'].permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)

            # Display
            axes[i].imshow(img)
            axes[i].set_title(f"Question: {sample['question']}\nPrediction: {sample['pred']}\nGround Truth: {sample['label']}")
            axes[i].axis('off')

        plt.tight_layout()
        status = "correct" if correct else "incorrect"
        plt.savefig(f"{args.save_path}/{status}_predictions.png")
        print(f"Visualization of {status} predictions saved to {args.save_path}/{status}_predictions.png")

    # Visualize correct predictions
    visualize_predictions(test_loader, model, device, idx_to_ans, num_samples=5, correct=True)

    # Visualize incorrect predictions (error cases)
    visualize_predictions(test_loader, model, device, idx_to_ans, num_samples=5, correct=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VQA model with frozen ResNet backbone")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference', 'train2'],
                        help='Mode to run the script in')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to the CLEVR dataset')
    parser.add_argument('--save_path', type=str, default='/kaggle/working',
                        help='Directory to save model checkpoints and outputs')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint (for inference mode)')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'train2':
        train2(args)
    elif args.mode == 'inference':
        if args.model_path is None:
            raise ValueError("Model path must be provided for inference mode")
        inference(args)