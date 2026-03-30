"""
Global constants for DLBT.

Latent ontology: K=16 states = 2^4, one per combination of four binary dimensions.

Bit layout (MSB to LSB):
  bit 3  front_back   0=front (y < 0.5),  1=back (y >= 0.5)
  bit 2  shape        0=triangular-faced, 1=non-triangular-faced
  bit 1  transp       0=not transparent,  1=transparent (t >= 0.5)
  bit 0  gloss        0=not glossy,       1=glossy (gl >= 0.5)

So latent_state = front_back*8 + shape*4 + transp*2 + gloss,
and index k in [0, 15].
"""

K: int = 16  # |Z| = 2^4

# Bit positions
DIM_FRONT_BACK = 3
DIM_SHAPE      = 2
DIM_TRANSP     = 1
DIM_GLOSS      = 0

# Binarisation thresholds (applied to raw latent values in [0,1] / world units)
Y_THRESHOLD    = 0.5   # y >= 0.5 (world units) -> back; y < 0.5 -> front
TRANSP_THRESH  = 0.5   # transparency in [0,1]
GLOSS_THRESH   = 0.5   # glossiness in [0,1]

# Shape categorisation
NON_TRIANGULAR_SHAPES = frozenset({"cube", "dodecahedron"})
# triangular-faced: tetrahedron (4 faces), octahedron (8), icosahedron (20)
