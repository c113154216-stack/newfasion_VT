# SMPL-X 模型權重放這裡

`body_model.load_smplx_body()` 會從這個資料夾載入 SMPL-X 權重檔。

這些檔案受官方授權條款保護，**無法自動下載**，需要你本人：

1. 到 https://smpl-x.is.tue.mpg.de/ 註冊帳號
2. 登入後同意授權條款（Model License），下載 `models_smplx_v1_1.zip`（或最新版本）
3. 解壓縮後把 `models/smplx/SMPLX_NEUTRAL.npz`（以及需要的話 `SMPLX_MALE.npz` / `SMPLX_FEMALE.npz`）放進這個資料夾，變成：
   ```
   assets/smplx/SMPLX_NEUTRAL.npz
   ```

在檔案就緒之前，程式會用 `body_model.build_proxy_mannequin()` 這個純幾何的替代人偶測試對齊邏輯，不會卡住開發進度。
