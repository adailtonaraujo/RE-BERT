# Train Data

In order to train your own software requirements extraction model, a set of software revision sentences previously annotated is required, pointing out parts of the text that are software requirements.

>**original sentence**: *The app crashes when I try to share photos*

>**software requirements of sentence**: *share photos*

These data must be used to generate a file with training data in **BIO format (short for Beginning, Inside, Outside)**. For each token (following the order of occurrence in the sentence), three lines will be created in the training file, where:

**Line 1**: original text of the sentence with the replacement of the current token by the marker $T$.
>$T$ app crashes when I try to share photos

**Line 2**: current token, replaced by the marker on Line 1.
>The

**Line 3**: (-1) if the current token is not part of the token sequence of a software requirement, (0) if the current token is the starting token of the token sequence of a software requirement or (1) if the current token is part of the software requirement but is not the initial token.
>-1

Complete sequence of lines in the training file for the token sequence of the review:

>$T$ app crashes when I try to share photos

>The

>-1

>The $T$ crashes when I try to share photos

>app

>-1

>The app $T$ when I try to share photos

>crashes

>-1

>The app crashes $T$ I try to share photos

>when

>-1

>The app crashes when $T$ try to share photos

>I

>-1

>The app crashes when I $T$ to share photos

>try

>-1

>The app crashes when I try $T$ share photos

>to

>-1

>The app crashes when I try to $T$ photos

>share

>0

>The app crashes when I try to share $T$

>photos

>1

In addition, include a record for the entire sequence of requirement tokens (required only when the requirement consists of more than one token).

>The app crashes when I try to $T$

>share photos

>1

On the next line, start the same process for the next review.

In the datasets_iob repository, all files that start with **train_iob** are in the IOB standard described earlier.


# Test Data

Files that start with **test_data** are data used for testing, each line of the file is the original text of a revision sentence. 

The files that start with **test_label** are the software requirements extracted from the sentences of the revisions of the file test_data, each line of the file has the requirements extracted from the sentence separated by ";". For sentences without software requirements the line is empty.
