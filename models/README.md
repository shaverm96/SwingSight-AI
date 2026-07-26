# Models

Most trained weights in this directory remain local and are ignored by Git.
The five-way club-type checkpoint is versioned so the project includes a
reference model for the broad club decision.

## Five-way club-type checkpoints

- `trained/club_type_5way.pt` — MobileNetV3-Small checkpoint for `driver`,
  `wood`, `hybrid`, `iron`, and `wedge`
- `trained/club_type_5way_cnn.pt` — compact custom-CNN baseline for the same
  five classes

Legacy broad/wood checkpoints are still supported for installations that do
not use the five-way model:

- `trained/club_broad_cnn.pt` — `iron` versus `wood`
- `trained/club_iron_number_cnn.pt` — legacy Iron numbers
- `trained/club_wood_type_cnn.pt` — `driver`, `wood`, or `hybrid`

## Exact Iron and Wedge markings

Exact markings do not use a project-trained checkpoint. Once the five-way
model returns Iron or Wedge, SwingSight invokes pretrained RapidOCR PP-OCR
models through ONNX Runtime on the full club image. The optional runtime is
declared in `requirements.txt`; no extra marking weights belong in this folder.

The reader accepts only valid Iron numbers, wedge abbreviations, or supported
wedge lofts after OCR confidence and normalization checks. See the root
`README.md` and `config.example.yaml` for configuration and failure behavior.
