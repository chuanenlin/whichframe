# Which Frame?

Search a video **semantically** with AI. For example, try a natural language search query like "a person with sunglasses". You can also search with images like Google's reverse image search and also a combined text + image. The underlying querying is powered by OpenAI’s CLIP neural network for "zero-shot" image classification.

---

## Try it out!

http://whichframe.chuanenlin.com

---

## Setting up

1.  Clone the repository.

```python
git clone https://github.com/chuanenlin/whichframe.git
cd whichframe
```

2.  Install package dependencies.

```python
pip install -r requirements.txt
```

3.  Run the app.

```python
streamlit run whichframe.py
```

---

## Examples

### 🔤 Text Search

#### Query

"three red cars side by side"

#### Result

![three-red-cars-side-by-side](/Users/david/Home/CMU/Notes/images/three-red-cars-side-by-side-6159437.jpeg)

### 🌅 Image Search

#### Query

![police-car-query](/Users/david/Home/CMU/Notes/images/helicopter-query.jpeg)

#### Result

![police-car-result](/Users/david/Home/CMU/Notes/images/helicopter-result.jpeg)

### 🔤 Text + 🌅 Image Search

#### Query

"a red subaru" +

![police-car-query](/Users/david/Home/CMU/Notes/images/police-car-query-6159437.jpeg)

#### Result

![subaru-and-police-car-result](/Users/david/Home/CMU/Notes/images/subaru-and-police-car-result-6159437.jpeg)
