# HAVEN

<p align="center">
  <a href="">Website</a> •
  <a href="">Paper</a> •
  <a href="./Data/test_data.json">Data</a> 
</p>

## 📰 News
- **2025-03-20**: We have released 1,200 questions and evaluation code.



## 👋 Overview
![Local Image](./Fig/Main.png)


### 🧐 Why HAVEN?
Previous research on hallucinations of LMMs has primarily focused on image understanding, as earlier LMMs could not process video inputs. These benchmarks are designed to evaluate hallucinations involving factors such as objects, relationships and attributes in a single image.

With advancements in multi-modal technologies, numerous LMMs now support video processing. Although many of these models did not incorporate audio inputs from videos, most can effectively process the visual content of video. Unlike image understanding, videos consist of sequences of multiple image frames over time, making video understanding more complex. It requires the analysis of continuous temporal dynamics, including sequential changes in human actions, object movements, and scene transitions. Hence, hallucinations in video understanding also differ from those in images.

To address the concern above, we proposed a benchmark for **HAllucination in Video UndErstaNding (HAVEN)**. HAVEN is meticulously designed to quantitatively evaluate the hallucination in video understanding for LMMs, which is constructed based on the following dimensions:

- Three **causes** of hallucinations: conflict with prior knowledge, in-context conflict, and inherent capability deficiencies of LMMs.
- Three types of hallucination **aspects** in a video: object, scene, and event.
- Three **formats** of questions: binary-choice, multiple-choice, and short-answer.


### 📈Data
Our video data comprises videos from three public video datasets ([`COIN`](https://coin-dataset.github.io/), [`ActivityNet`](http://activity-net.org/download.html), and [`Sports1M`](https://github.com/gtoderici/sports-1m-dataset) and manually collected video clips from Internet. You can directly download the video from our provided source link.

Here is an example of our data:
```json
{
  "Index": 1,
  "Question": "Are the individuals in the video sewing garments with thread?",
  "Answer": "No",
  "Causes": "Conflict with prior",
  "Aspects": "Event",
  "Form": "Binary-choice",
  "Source Link": "https://www.youtube.com/embed/xZecGPPhbHE",
  "Begin": "0:29",
  "End": "0:40",
  "Video Path": "Coin0001",
  "Group_id": "Coin_group0001"
}
```

### Explanation of Fields:  
- **Index**: The global question index, indicating the sequential order of all benchmark questions.  
- **Causes**: Represents the cause of hallucination, corresponding to the categories introduced in the paper:  
  - *Conflict with prior knowledge*: The model generates answers that contradict well-established knowledge.  
  - *In-context conflict*: The model produces contradictory answers within the given video context.  
  - *Capability deficiency*: The model lacks the ability to recognize or reason about the given video content.  
- **Aspects**: Defines the hallucination category based on video content:  
  - *Object*: Hallucinations related to incorrect object recognition.  
  - *Scene*: Misinterpretations of scene context.  
  - *Event*: Incorrect understanding of events or actions within the video.  
- **Form**: Represents the question format, aligning with the paper's categorization:  
  - *Binary-choice*: A question with two answer choices (e.g., True/False, Yes/No).  
  - *Multiple-choice*: A question with multiple answer choices, requiring the selection of the most appropriate one.  
  - *Short-answer*: A question requiring a direct response without predefined options.  
- **Source Link**: The original video file path. If the video is from ActivityNet, it can be downloaded from **[http://activity-net.org/download.html]**.  
- **Begin, End**: Indicate the start and end timestamps of the relevant video segment.  
- **Video Path**: The unique identifier for the video. The same Video Path across multiple entries means they reference the same video clip.  
- **Group_id**: A group identifier for related questions, used to analyze model consistency across different questions related to the same content. 
  - Identical `group_id` values indicate different variations of the same question.  
  - The prefix of `group_id` indicates the dataset from which the question originates. For example:  
    - `coin_groupXXXX` refers to the *COIN* dataset.  
    - `Sports1M_groupXXXX` refers to the *Sports1M* dataset.  
    - `ActivityNet_groupXXXX` refers to the *ActivityNet* dataset.  
    - `YouTube_groupXXXX` refers to videos on *YouTube*. 



## 🚀 Quickstart



## 🙇‍♂️ Acknowledgement



## ✍️ Citation
If you find our work helpful, please cite as
```

```

