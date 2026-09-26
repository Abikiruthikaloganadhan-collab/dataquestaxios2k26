Task A — Conversation Structure Reconstruction
Objective
Task A aims to reconstruct the structure of interleaved group-chat conversations.

For every message, the system predicts:

reply_to_message_id — the earlier message that the current message is replying to, or NONE if it starts a new discussion.

thread_id — the discussion/thread to which the message belongs.

The challenge is that messages are provided in timestamp order, while multiple independent discussions can be interleaved within the same channel. Therefore, the previous message is not necessarily the message being answered.

Input
The model is trained using taskA_train.csv.

The training data contains:

episode_id

message_id

sender

timestamp

text

reply_to_message_id

thread_id

The hidden/public evaluation file contains the message information but does not provide the two target columns.

Approach
We formulate Task A as an earlier-message candidate selection problem.

For each current message:

Consider only earlier messages from the same episode as possible reply candidates.

Calculate features describing how well each candidate matches the current message.

Use a Logistic Regression classifier to estimate the probability that each candidate is the correct parent.

Select the highest-probability candidate.

If the highest probability is below the confidence threshold, predict NONE.

Reconstruct discussion threads from the predicted reply relationships.

Features
The model uses the following features:

1. TF-IDF text similarity
TF-IDF converts message text into numerical vectors. Similar wording between a current message and a candidate reply produces a higher similarity score.

Both single-word and two-word combinations are considered.

2. Word overlap
We calculate the proportion of shared words between the current message and the candidate message.

3. Timestamp gap
The time difference between the current message and the candidate is used as a supporting conversational signal.

4. Message-position distance
We calculate how many message positions separate the current message from the candidate.

5. Sender compatibility
We check whether the current message and candidate have the same sender. This provides information about turn-taking patterns.

Model
A Logistic Regression classifier is trained using the labeled training episodes.

For each pair:

current message + earlier candidate message
the model learns whether that candidate is the correct parent.

During prediction, every earlier candidate receives a probability. The candidate with the highest probability is selected as the predicted reply.

Thread Reconstruction
After predicting the reply relationships, the system reconstructs the discussion structure.

For example:

M1 → NONE
M2 → M1
M3 → NONE
M4 → M3
M5 → M2
The resulting threads are:

Thread 1: M1 → M2 → M5
Thread 2: M3 → M4
Messages that ultimately point to the same root message receive the same thread_id.

Output
The final Task A submission is:

submission_taskA.csv
with the required columns:

episode_id,message_id,reply_to_message_id,thread_id
Example:

episode_id,message_id,reply_to_message_id,thread_id
A002,A002-M001,NONE,A002-T1
A002,A002-M002,A002-M001,A002-T1
A002,A002-M005,NONE,A002-T2
Validation
Before saving the submission, the code checks that:

every hidden message appears exactly once;

there are no duplicate message rows;

every predicted reply is either NONE or an earlier message;

the output contains the required columns.

Limitations
The approach can struggle when:

a reply has little textual similarity to its parent;

different discussions use similar vocabulary;

a reply occurs much later than its parent;

multiple earlier messages are similarly plausible candidates.

These cases are difficult because the visible message order does not directly reveal the underlying discussion structure.

Summary
The Task A pipeline is:

Training data
      ↓
Generate earlier-message candidates
      ↓
Extract text + structural features
      ↓
Train Logistic Regression
      ↓
Score candidate replies
      ↓
Predict reply_to_message_id
      ↓
Follow reply links
      ↓
Construct thread_id
      ↓
submission_taskA.csv
The main idea is to treat conversation reconstruction as a candidate matching problem, combining textual and conversational signals to recover the hidden discussion structure.

## Task B — Deleted Message Reconstruction

### Problem

Task B focuses on reconstructing deleted messages from conversation episodes.

Some messages are removed from a conversation, and there is no explicit marker showing the missing position. For each missing position (`gap_id`), the dataset provides multiple candidate messages. Exactly one candidate corresponds to the deleted message.

The objective is to **rank the candidate messages from most likely to least likely** for every gap.

### Dataset

Two public datasets are used:

#### 1. `taskB_messages_public.csv`

Contains the remaining messages from the conversation.

Main columns:

* `episode_id`
* `message_id`
* `sender`
* `timestamp`
* `text`

This dataset provides the conversational context.

#### 2. `taskB_candidates_public.csv`

Contains the possible candidates for each missing message.

Important columns:

* `episode_id`
* `gap_id`
* `gap_position`
* `candidate_id`
* `candidate_text`
* `candidate_sender`
* `candidate_timestamp`
* `feature_sender_pattern_fit`
* `feature_timestamp_gap_fit`
* `feature_thread_position_fit`
* `feature_embedding_similarity`

### Approach

Our solution combines the four provided candidate-fit features into a single weighted score:

```text
Final Score =
0.20 × Sender Pattern Fit
+ 0.15 × Timestamp Gap Fit
+ 0.30 × Thread Position Fit
+ 0.35 × Embedding Similarity
```

The candidates are then sorted by their final score within each `gap_id`.

The highest-scoring candidate receives rank `1`, followed by rank `2`, rank `3`, and so on.

### Workflow

```text
Conversation + Candidate Messages
              ↓
      Four Candidate Features
              ↓
        Weighted Scoring
              ↓
        Final Candidate Score
              ↓
    Ranking within each Gap
              ↓
       Final Submission CSV
```

### Experiments

We also tested additional approaches during development.

**Normalization:**
Feature values were normalized within each gap to make their relative scales comparable. The resulting candidate rankings did not change significantly, so normalization was not retained as a separate component of the final method.

**TF-IDF Context Similarity:**
We experimented with TF-IDF and cosine similarity to measure additional textual similarity between candidates and nearby conversation messages. The resulting scores were often very small or zero because TF-IDF relies on lexical overlap. Therefore, the provided embedding similarity was retained as the main semantic signal.

### Output

The final submission contains:

```text
gap
```
[submission_taskA.csv](https://github.com/user-attachments/files/32677330/submission_taskA.csv)
[submission_taskB.csv](https://github.com/user-attachments/files/32677323/submission_taskB.csv)

https://chatgpt.com/share/6ab761b2-4d04-83ee-b810-99936cf2b499
https://chatgpt.com/share/6ab761aa-ca74-83e8-918d-965b027821f7
