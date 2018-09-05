# Text generation using LSTMs

Long short term memory neural networks are trained on lord of rings dataset to predict the successive word.

This example does not generate character by character, rather goes word by word. Given first n words, the network predicts `n+1`th word. The network is trained with 10 input words and the successive word is given as output word.

After enough training of the model, the network is able to generate sensible sentences relating to the vocabulary as used in lord of rings. In order to obtain the following text, I have trained with 4xTeslaV100 GPU cores for 60 to 80 epochs approx.

### Generated text:

'Hobbits lived in the woods happily and the story begins to climb out of the North to be seen in the light of the old man. The Dark Lord were bent in the dark and the light of the great ship of the Road and the night and the dwarves were drawing off to the ground and the grass and the main door was gone. The foremost were flung with candles and as the Sun and the horns leapt from the hill and came out of the mist and the sun was in the sky and the sun was shining and a white figure ran down into a wide shallow sky above them and they were in the air. There was a great deal of ancient and smooth and the mountains was coming. Now there were no sign of the Enemy. There was a ford in the great light of the great range of Dale and the Riders of the Eagles and on the borders of the Shire and the great mountains was thrown down and the road and the River was in the East and the travellers was shut. The men were already in a cloud of gold. There was no longer. He could not see him to see'