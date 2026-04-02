"""
Global constants for DLBT.

Latent ontology: K=16 states = 2^4, one per combination of four binary dimensions.

Bit layout (MSB to LSB):
  bit 3  left_right    0=left   (x < 0.0),    1=right (x >= 0.0)
  bit 2  transp        0=opaque (t < 0.5),     1=transparent (t >= 0.5)
  bit 1  gloss         0=matte  (gl < 0.5),    1=glossy (gl >= 0.5)
  bit 0  small_large   0=small  (s < 0.63),    1=large (s >= 0.63)

So latent_state = left_right*8 + transp*4 + gloss*2 + small_large,
and index k in [0, 15].

Note: front/back (Y depth) is excluded — it confounds with apparent size
in perspective rendering. Y position is randomised in the stimuli but is
not a modelled latent dimension.
"""

K: int = 16  # |Z| = 2^4

# Bit positions
DIM_LEFT_RIGHT  = 3
DIM_TRANSP      = 2
DIM_GLOSS       = 1
DIM_SMALL_LARGE = 0

# Binarisation thresholds
X_THRESHOLD   = 0.0    # x >= 0.0 -> right; x < 0.0 -> left
TRANSP_THRESH = 0.5    # transparency in [0,1]
GLOSS_THRESH  = 0.5    # glossiness in [0,1]
SCALE_THRESH  = 0.63   # scale >= 0.63 -> large  (midpoint of obj_scale_range [0.38, 0.88])
