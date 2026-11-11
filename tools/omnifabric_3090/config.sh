# go.sh 的設定：第一次改這幾行就好，之後不用再動。路徑可以用 ~。
OMNI_DIR=~/OmniFabric              # OmniFabric 裝在哪
HF_HOME=~/hf_cache                 # FLUX 等模型的快取，要有約 160 GB 空位
PHOTOS=$HERE/photos               # 衣服照片（預設是這個資料夾裡的 photos/）：<衣服>_front.png、<衣服>_back.png（背面可以沒有）
OUT=~/omnifabric_out               # 輸出；要帶回來的是 $OUT/collect/
CHATGARMENT_BIN=                   # ChatGarment 的 pytorch_model.bin；空白就自動在家目錄找
GARMENTS=upper                     # upper＝只做上衣；all＝照片裡預測出的全部
