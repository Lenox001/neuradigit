## 📄 README.md

```markdown
# NeuraDigit

A beginner-friendly AI that recognises handwritten digits (0–9) using a feed‑forward neural network built with PyTorch.  
It’s trained on the classic MNIST dataset and achieves about **97% accuracy** on unseen test images.

---

## 🧠 What It Does

- Learns from 60,000 28×28 grayscale images of handwritten digits  
- Predicts the digit in any new 28×28 image  
- Serves as a gentle introduction to deep learning, data loading, training loops, and model saving

---

## 📂 Project Structure

```
NeuraDigit/
├── neuradigit.py        # main script: training + evaluation
├── mnist_model.pth      # saved model weights (after training)
├── data/                # MNIST dataset (downloaded automatically)
└── README.md            # this file
```

---

## ⚙️ Requirements

- Python 3.8 or newer  
- PyTorch (CPU version works fine)  
- torchvision  
- matplotlib (optional, for viewing samples)

Install everything with:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib
```

*If you have an NVIDIA GPU and want faster training, use the appropriate CUDA index from [pytorch.org](https://pytorch.org).*

---

## 🚀 Getting Started

1. **Clone or download** this repository.
2. **Install dependencies** (see above).
3. **Run the script**:

   ```bash
   python neuradigit.py
   ```

   The first run will download the MNIST dataset (about 12 MB) into a `data/` folder.  
   After that, training starts automatically.

---

## 🏋️ Training

The script trains for 5 epochs (you can change `num_epochs` inside the code).  
You’ll see output like:

```
Epoch [1/5], Step [100/600], Loss: 0.4521
Epoch [1/5], Step [200/600], Loss: 0.2876
...
Accuracy on the 10000 test images: 97.43%
Model saved as mnist_model.pth
```

---

## 🧪 Testing

After training, the model is evaluated on 10,000 test images.  
No extra action is needed — the script prints the final accuracy automatically.

---

## 💾 Using the Saved Model

The trained weights are saved to `mnist_model.pth`. To load them later for prediction:

```python
import torch
from neuradigit import SimpleNN, input_size, hidden_size, num_classes, device

model = SimpleNN(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load('mnist_model.pth'))
model.to(device)
model.eval()

# Now use model(processed_image) to predict a digit
```

---

## 🛠 Customisation

- **Change the architecture**: edit the `SimpleNN` class (add layers, change hidden size, etc.)
- **Train on a GPU**: the code automatically detects a CUDA-capable device. Just install the CUDA version of PyTorch.
- **Predict your own handwriting**: preprocess a 28×28 grayscale image into a tensor with pixel values normalised to [-1, 1] and call `model(image_tensor)`.

---

## 📖 Licence

This project is for educational purposes. Feel free to use, modify, and share it.

---

## 🙋‍♀️ Questions?

If you run into issues, check that:
- Python ≥3.8 is installed
- PyTorch installed without errors (`import torch; print(torch.__version__)`)
- The `data/` folder has write permissions (for the dataset download)
```

---