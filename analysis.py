import pandas as pd
import config

df = pd.read_csv(config.WIKIM_TEST_CLEAN, sep="\t", names=["p1", "p2", "words", "mentions", "types"])
pred = pd.read_csv("output/pred.tsv", sep="\t", names=["pred", "label"])

df = pd.concat([df, pred], axis=1)
mask1 = df.apply(lambda x: set(x["pred"].split("|")) != set(x["label"].split("|")), axis=1)
mask2 = df.types.map(lambda x: "person" in x)
print(sum(mask1), sum(mask2))
df[(mask1 & mask2)][["words", "mentions", "pred", "label"]].to_csv("output/analysis.tsv", sep="\t", header=False, index=False)
