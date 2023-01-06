# Import dependencies
import cv2
import torch
import matplotlib.pyplot as plt

# Loading model 
midas = torch.hub.load('intel-isl/MiDaS','MiDaS_small')
midas.to('cpu')
midas.eval()

# Input transformational pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

# OpenCV
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    # cv2.imshow('frame', frame)
    # Transoform input for midas
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cpu')

    # Making predictions
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size = img.shape[:2],
            mode = 'bicubic',
            align_corners=False,
        ).squeeze()

        output = prediction.cpu().numpy()
        print(output)
    plt.imshow(output)
    cv2.imshow('frame', frame)
    plt.pause(0.00001)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
plt.show()