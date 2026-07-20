# Models

Store trained weights here (not committed to GitHub).

The club-recognition CNN expects these checkpoint paths by default:

- `trained/club_broad_cnn.pt` — `iron` vs `wood`
- `trained/club_iron_number_cnn.pt` — `1` through `9`
- `trained/club_wood_type_cnn.pt` — `driver`, `wood`, or `hybrid`

Create them with `scripts/train_club_cnn.py`; each checkpoint stores the required task and class order, and inference validates both before using it.
