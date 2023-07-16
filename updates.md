## todo list

- i'm not taking into account the case in which the target is empty and the model is still predictin something, i am just discarding it. (it is ok, because the critical case is basically when the targect expects something but the faulty model doesn't predict anything)

- I'm now saving only boxes with a high confidence level
- I'm now saving only the boxes of the faulty model with a iou score lower than 0.8 or with a different label
- Also the iou score is saved for each bb
- Only the bbs with an iou score higher than 0.7 (for golden) and 0.6 (for faulty) are evaluated.

- Each record of F_*_results.csv is saved for each bounding box of the golden model
- should i consider a different file for the counters of bounding boxes? Notice that it is done by image