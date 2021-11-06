---
title: Which Frame?
emoji: 🔍
colorFrom: purple
colorTo: purple
sdk: streamlit
app_file: whichframe.py
---

# Which Frame?

🧠 Search a video **semantically** with OpenAI's CLIP neural network.

❓ Example: Which frame has a person with sunglasses and earphones?

🔍 Try searching with text, image, or text + image.

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

### 1. Text Search

#### Query

"three red cars side by side"

#### Result

![three-red-cars-side-by-side](examples/three-red-cars-side-by-side.jpeg)

### 2. Image Search

#### Query

![police-car-query](examples/helicopter-query.jpeg)

#### Result

![police-car-result](examples/helicopter-result.jpeg)

### 3. Text + Image Search

#### Query

"a red subaru" +

![police-car-query](examples/police-car-query.jpeg)

#### Result

![subaru-and-police-car-result](examples/subaru-and-police-car-result.jpeg)
