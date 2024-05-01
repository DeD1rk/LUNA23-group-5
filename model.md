# Multi-task model

```
(64,128,128 @ 1) \ <contraction blocks>
        (64,128,128 @ 16) \-   -   -   -   -   -   -   -   -   -   -   -   -   -  (64,128,128 @ 8) -> (64,128,128 @ 1) -> segmentation DICE loss
                (32,64,64 @ 32) \  -   -   -   -   -   -   -   -   -   -   -  (32,64,64 @ 16) /
                        (16,32,32 @ 64) \  -   -   -   -   -   -   -  (16,32,32 @ 32) /
                                (8,16,16 @ 64 ) \  -   -   -   - (8,16,16 @ 64) /
                                    (4,8,8 @ 128 ) \   -  (4,8,8 @ 128) /
                                        (2,4,4 @ 128 ) -   / <expansion blocks>
                                                           \ <1x1x1 conv>                              /-- (4) -> nodule type CCE loss
                                                            (2,4,4 @ 64) -<flatten>- (2048) -> (128) - 
                                                                                                       \-- (1) -> malignancy BCE loss 
```

Losses are combined by weighted sum: $$\mathcal{L}_{total} = w_{seg} \cdot \mathcal{L}_{seg} + w_{type} \cdot \mathcal{L}_{type} + w_{mal} \cdot \mathcal{L}_{mal}$$

We can tune the weights based on experimentation and the scales of the individual losses to get a good balance between the different tasks.