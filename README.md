# Convolutional Layer Visualizer

A simple, framework-free web app to help students understand how RGB images flow through a convolutional layer.

- No build step, no frameworks. Just open `index.html`.
- Split the input image into its R, G, B channels (shown as grayscale).
- Apply K 3×3×3 kernels (depth 3, one 3×3 per RGB) with stride 1 and optional same padding.
- Visualize the kernels (values and color-coded heat) and the K output feature maps.

## Features

- Load your own image (or click “Use sample” to generate a procedural test image).
- Choose the number of kernels (1–8).
- Pick a kernel set:
	- Random normalized weights (zero-mean, L1-normalized across all 27 weights)
	- Classic filters (edge, blur, sharpen, Sobel, etc.) replicated across RGB
- Uses valid convolution (no padding) and no ReLU for clarity.
- Per-map min/max shown for quick intuition; outputs normalized per map for display.

## How to run

You can open the app directly in your browser—no server required:

1. Open the file `index.html` in a modern browser.
2. Click “Use sample” or load your own image.
3. Adjust the number of kernels and options; the outputs update live.

Optional: If you prefer a local static server, any will do. For example, with Python installed:

```bash
python3 -m http.server 8787
```

Then visit http://localhost:8787 in your browser and open `index.html`.

## How it works (short)

For each kernel k, we compute a single output feature map as the sum of 2D convolutions over R, G, and B with that kernel’s per-channel 3×3 weights:

output_k = conv2d(R, k_R) + conv2d(G, k_G) + conv2d(B, k_B)

Stride = 1. Padding is Valid (no padding). No ReLU is applied. Outputs are normalized to [0, 255] for visualization.

## Files

- `index.html` — UI and containers for canvases and controls.
- `styles.css` — Minimal styling and layout, plus kernel heatmap color scale.
- `script.js` — Image loading, RGB split, convolution logic, kernel generation, and rendering.

## Notes

- The “Classic filters” use the same 3×3 filter on all three color channels to keep the concept clear.
- Random kernels are zero-mean and L1-normalized across all 27 values so magnitudes are comparable.
- Outputs are per-map normalized for display—absolute scale is not preserved visually.

## License

MIT