

- i'm not taking into account the case in which the target is empty and the model is still predictin something, i am just discarding it. (it is ok, because the critical case is basically when the targect expects something but the faulty model doesn't predict anything)

- I'm now saving only boxes with a high confidence level
- I'm now saving only the boxes of the faulty model with a iou score lower than 0.8 or with a different label
- Also the iou score is saved for each bb
- Only the bbs with an iou score higher than 0.7 (for golden) and 0.6 (for faulty) are evaluated.

- Each record of F_*_results.csv is saved for each bounding box of the golden model
- should i consider a different file for the counters of bounding boxes? Notice that it is done by image


- Far ripartire le diverse simulazioni con altri layer

- Quando i csv saranno completi, voglio estrarre solo le bitmask corrispondenti al bit 30
- vedere cosa si può fare
- Cercare di giustificare perchè è quello il punto più critico

- Rigenerare tutti i grafici

- dare un senso al clustering associandolo ad ogni configurazione.


- come funziona la fault injection? e che risultato mi aspetto sui pesi?
- come dovrebbero essere cambiati?
- cercare un paper che descrive come dovrebbero essere i pesi di una rete neurale che funziona bene e una che funziona male.



Avendo fatto delle ricerche ho scoperto che dalla distribuzione dei pesi difficilmente si riesce a capire come la rete viene impattata. Di base, se la rete ha certi risultati con una specifica distribuzione di pesi, con un'altra distribuzione la rete farà cose diverse. Vale la pena in ogni caso andare a vedere  come cambia la distribuzione. 

## todo list
- guardare NVbit 