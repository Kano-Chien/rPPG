# Real-time System for Remote Heart Rate Measurement
This system is designed for remote heart rate measurement, utilizing a deep learning model to capture signals from our face and converting them into heart rate readings.
## Introduction
rPPG is a non-contact heart rate measurement technology based on an RGB camera. Its principle involves utilizing a deep learning model to capture subtle color variations on the face and convert them into heart rate information. During heart contractions, there are slight color changes on the face. Traditional contact-based heart rate monitoring devices utilize PPG signals to derive heart rate information. PPG signals involve emitting light onto the skin and measuring the changes in light absorption to achieve heart rate measurement. Our deep learning model is trained using actual collected PPG signals as samples to learn the biological signals of heart rate on the face.

<p align="center">
  <img src="https://github.com/jiaan40504/rPPG/blob/main/rppg/image/rppg.png?raw=true">
</p>

## Motivation
Traditional heart rate monitoring devices are contact-based, requiring the device to be in contact with the patient's skin to measure PPG signals and then convert them into heart rate information. Over time, this can lead to patient discomfort. Therefore, utilizing non-contact devices for remote heart rate monitoring is highly important.

rPPG signals are easily affected by several factors: (1) camera noise, (2) slight movements of the face, (3) changes in lighting conditions and different application scenarios. Under the influence of these variables, traditional methods are no longer sufficient. In recent years, many researchers have turned to deep learning models to address this issue, such as ETA, HRCNN, etc. Among them, Siamese-rPPG has demonstrated excellent performance [1]. Therefore, we referenced the architecture of Siamese-rPPG and made improvements. Since videos contain redundant signals, they used a 3DCNN to capture the features of the temporal and facial rPPG signals. Based on this network, we employed non-overlapping and spatial attention in real-time, and by utilizing FFT transformation, we achieved real-time heart rate results for remote heart rate monitoring.

## Network Architecture
<p align="center">
  <img src="https://github.com/jiaan40504/rPPG/blob/main/rppg/image/Network%20Architecture.png?raw=true">
</p>

## Loss Function
<p align="center">
  <img src="https://github.com/jiaan40504/rPPG/blob/main/rppg/image/Pearson%20correlation%20Loss.png?raw=true">
</p>

## Result
<p align="center">
  <img src="https://github.com/jiaan40504/rPPG/blob/main/rppg/image/Comparison.png?raw=true">
</p>
