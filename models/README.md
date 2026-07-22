# Models

Most trained weights in this directory remain local and are ignored by Git.
The two five-way club-type checkpoints below are intentionally versioned so
the repository includes a usable reference model.

The club-recognition CNN expects these checkpoint paths by default:

- `trained/club_broad_cnn.pt` — `iron` vs `wood`
- `trained/club_iron_number_cnn.pt` — `1` through `9`
- `trained/club_wood_type_cnn.pt` — `driver`, `wood`, or `hybrid`

Five-way club-type checkpoints:

- `trained/club_type_5way.pt` — MobileNetV3-Small checkpoint for `driver`,
	`wood`, `hybrid`, `iron`, and `wedge`
- `trained/club_type_5way_cnn.pt` — compact custom-CNN baseline for the same
	five classes

Create them with `scripts/train_club_cnn.py`; each checkpoint stores the required task and class order, and inference validates both before using it.
