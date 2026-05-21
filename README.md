# lung-sound-classifier
Upload an MP3 lung sound. The tool splits it into segments, converts each one into a spectrogram image, and runs it through EfficientNet-B0. It averages the predictions across all segments to give one final result from 9 conditions  including Pneumonia, COVID-19, and Tuberculosis.
