# Image Captioning using Pytorch
Current Implementation -> 
[Show, Attend, and Tell](https://arxiv.org/abs/1502.03044)

### Papers' results

![](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/ebd0cc84-dc41-455a-8ef3-186afab3eba1/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210521%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210521T211413Z&X-Amz-Expires=86400&X-Amz-Signature=3a57eaaaf1c9e6c1057656d612bfb21867ac10b9f1fff8acc2924133b5afb988&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

- The Goggle NIC was the first implementation from the paper "Show and Tell"
- Soft-Attention is the result I'm comparing to. I'm not comparing to Hard-Attention as it's trainable by maximizing an approximate vairational lower bound (REINFORCE) while Soft-Attention is trainable by standard back-propagation.

### My Best Results (top bleu-4)

- using a beam size of 3:

    ![](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/098019da-5e7f-4123-8d71-1fb9b1304683/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210521%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210521T211433Z&X-Amz-Expires=86400&X-Amz-Signature=6f20adc250bb63a2d89063c71ff629c598d7831aa8626e670139bf4960c6c7b3&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

    - Bleu-1 is lower than the results of the paper, this maybe due to one or all of the below reasons:
        - We are using a smaller vocabulary size so the model has less number of words in its knowledge.
        - While training, validation and also testing we are trying to get the highest bleu-4 so we ignored higher bleu-1 before because it has lower bleu-4.
            - The beam search also finds the sequences with the highest score which depends on the loss function.
            - And we chose the best model based on bleu-4 only.
    - The other bleu scores are higher as they depend more on the pairs of words which is our main optimization goal

[Implementaiton Differences Table](https://www.notion.so/a168d58b9e194572bb33d232bd782a8d)

