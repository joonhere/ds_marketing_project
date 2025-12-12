import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image

pubmed_title = pd.read_csv("pubmed_title.csv")
pubmed_title.head()

plt.figure(figsize=(10, 5))

text = str(list(pubmed_title['Title']))
mask = np.array(Image.open('image.jpg'))
cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("", ['#000066','#003399', '#00FFFF'])


wordcloud = WordCloud(background_color = 'white', width = 2500,  height = 1400,
                      max_words = 170, mask = mask, colormap=cmap).generate(text)


plt.imshow(wordcloud)
plt.axis('off')

plt.suptitle('Heart Disease Wordcloud', fontweight='bold', fontfamily='serif', fontsize=15)
plt.title('Title of abstract in Pubmed site: Heart Failure', fontfamily='serif', fontsize=12)
plt.show()