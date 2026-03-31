"""
Global constants for DLBT.

Latent ontology: K=32 states = 2^5, one per combination of five binary dimensions.

Bit layout (MSB to LSB):
  bit 4  front_back    0=front  (y < 0.5),    1=back  (y >= 0.5)
  bit 3  left_right    0=left   (x < 0.0),    1=right (x >= 0.0)
  bit 2  transp        0=opaque (t < 0.5),     1=transparent (t >= 0.5)
  bit 1  gloss         0=matte  (gl < 0.5),    1=glossy (gl >= 0.5)
  bit 0  small_large   0=small  (s < 0.65),    1=large (s >= 0.65)

So latent_state = front_back*16 + left_right*8 + transp*4 + gloss*2 + small_large,
and index k in [0, 31].
"""

K: int = 32  # |Z| = 2^5

# Bit positions
DIM_FRONT_BACK  = 4
DIM_LEFT_RIGHT  = 3
DIM_TRANSP      = 2
DIM_GLOSS       = 1
DIM_SMALL_LARGE = 0

# Binarisation thresholds (applied to raw latent values)
Y_THRESHOLD   = 0.5    # y >= 0.5 (world units) -> back; y < 0.5 -> front
X_THRESHOLD   = 0.0    # x >= 0.0 (world units) -> right; x < 0.0 -> left
TRANSP_THRESH = 0.5    # transparency in [0,1]
GLOSS_THRESH  = 0.5    # glossiness in [0,1]
SCALE_THRESH  = 0.65   # scale >= 0.65 -> large; scale < 0.65 -> small
