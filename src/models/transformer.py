"""DistilBERT fine-tuning for document classification."""

from pathlib import Path

import numpy as np
import structlog
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.data.dataset import ComplaintDataset
from src.utils.config import load_config

logger = structlog.get_logger(__name__)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TransformerClassifier:
    """Fine-tuned DistilBERT for document classification."""

    def __init__(
        self,
        num_classes: int,
        label_names: list[str],
        config_path: str = "model_config.yaml",
    ):
        self.config = load_config(config_path)
        self.tf_config = self.config["transformer"]
        self.num_classes = num_classes
        self.label_names = label_names
        self.device = get_device()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tf_config["model_name"]
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.tf_config["model_name"],
            num_labels=num_classes,
        ).to(self.device)

        logger.info(
            "Transformer initialized",
            model=self.tf_config["model_name"],
            device=str(self.device),
            num_classes=num_classes,
        )

    def train(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: list[str],
        val_labels: list[int],
    ) -> dict:
        """Fine-tune the transformer model."""
        train_dataset = ComplaintDataset(
            train_texts, train_labels, self.tokenizer, self.tf_config["max_length"]
        )
        val_dataset = ComplaintDataset(
            val_texts, val_labels, self.tokenizer, self.tf_config["max_length"]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.tf_config["batch_size"],
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.tf_config["batch_size"] * 2,
            shuffle=False,
            num_workers=0,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.tf_config["learning_rate"],
            weight_decay=self.tf_config["weight_decay"],
        )

        total_steps = len(train_loader) * self.tf_config["epochs"]
        warmup_steps = int(total_steps * self.tf_config["warmup_ratio"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )

        best_val_f1 = 0.0
        best_state = None
        history = {"train_loss": [], "val_loss": [], "val_f1": []}

        for epoch in range(self.tf_config["epochs"]):
            # Training
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                n_batches += 1

            avg_train_loss = total_loss / n_batches

            # Validation
            val_loss, val_f1 = self._evaluate(val_loader)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)

            logger.info(
                "Epoch complete",
                epoch=epoch + 1,
                train_loss=f"{avg_train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                val_f1=f"{val_f1:.4f}",
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        logger.info("Training complete", best_val_f1=f"{best_val_f1:.4f}")
        return history

    def _evaluate(self, dataloader: DataLoader) -> tuple[float, float]:
        """Evaluate the model on a dataloader."""
        from sklearn.metrics import f1_score

        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                total_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        f1 = f1_score(all_labels, all_preds, average="macro")
        return avg_loss, f1

    def predict(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Predict classes and probabilities for texts."""
        dataset = ComplaintDataset(
            texts, None, self.tokenizer, self.tf_config["max_length"]
        )
        loader = DataLoader(
            dataset, batch_size=self.tf_config["batch_size"] * 2, shuffle=False
        )

        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                probs = torch.softmax(outputs.logits, dim=-1)
                all_probs.extend(probs.cpu().numpy())

        probabilities = np.array(all_probs)
        predictions = np.argmax(probabilities, axis=1)
        return predictions, probabilities

    def predict_single(self, text: str) -> tuple[str, float, dict[str, float]]:
        """Predict class for a single text with probabilities per class."""
        preds, probs = self.predict([text])
        pred_idx = preds[0]
        pred_label = self.label_names[pred_idx]
        confidence = float(probs[0][pred_idx])
        class_probs = {
            name: float(probs[0][i]) for i, name in enumerate(self.label_names)
        }
        return pred_label, confidence, class_probs

    def save(self, path: Path) -> None:
        """Save model and tokenizer."""
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path / "model")
        self.tokenizer.save_pretrained(path / "tokenizer")

        import json

        with open(path / "label_names.json", "w") as f:
            json.dump(self.label_names, f)

        logger.info("Transformer saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> "TransformerClassifier":
        """Load a saved transformer classifier."""
        import json

        with open(path / "label_names.json") as f:
            label_names = json.load(f)

        instance = cls.__new__(cls)
        instance.device = get_device()
        instance.label_names = label_names
        instance.num_classes = len(label_names)
        instance.config = load_config("model_config.yaml")
        instance.tf_config = instance.config["transformer"]

        instance.tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
        instance.model = AutoModelForSequenceClassification.from_pretrained(
            path / "model"
        ).to(instance.device)

        logger.info("Transformer loaded", path=str(path))
        return instance
