Zhuyin convert to Chinese Word
===
This Project is made with ELMo and BiLSTM.
## Requirement
- download the ELMo model from [here](https://www.dropbox.com/s/kqohf7sypkb26pc/ELMoForManyLangs.zip?dl=1) to the main directory.

## Usage
```
$ python zhuyin2char.py live
model.ckpt load!
sent> ㄉㄚˋ ㄐㄧㄚ ㄏㄠˇ
--
大 家 好
```

When running demo, punctuations or English words or numbers are not allow, otherwise it will output some wrong words.
