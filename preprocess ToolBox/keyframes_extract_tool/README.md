# Usage example:

```
from keyframes_extract_diff import KeyFrames
keyframes = KeyFrames() 
keyframes = KeyFrames(mode=KeyFrames.USE_LOCAL_MAXIMA | KeyFrames.USE_THRESH | KeyFrames.USE_TOP_ORDER, debug = True, thresh=0.7, len_window=60 )
keyframes.extract("pikachu.mp4") 
keyframes.extract("pikachu.mp4", mode=KeyFrames.USE_THRESH, debug = False)
```
