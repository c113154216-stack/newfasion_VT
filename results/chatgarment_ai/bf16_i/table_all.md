# bf16_i 總表（50 件）

由 `tools/chatgarment/experiments/batch_table.py` 產生。參數是 `to_design.py` 修正後的（款式修正已套用，衣長、領深校正還沒套用）。

- **衣長**：`shirt.length`（GarmentCode 預設 1.2）；**寬鬆**：`shirt.width`（1.0～1.3）；**袖長**：`sleeve.length`（1＝到手腕附近）
- **描述有袖口**：AI 描述的 extra 有寫 cuffs（例如 ribbed cuffs）；**袖口參數**：`sleeve.cuff.type`

| 衣服 | 正確領型 | AI 名稱 | 上衣 | 開襟 | 衣長 | 寬鬆 | AI 描述的領子 | 領型＋配件 | AI 描述的袖子 | 袖長 | 描述有袖口 | 袖口參數 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| i01_blue_crew_tee | 圓領 | T-shirt | 一般 |  | 1.23 | 1.05 | crew neck | 圓＋－ | short sleeves, normal sleeves | 0.16 |  | － |
| i02_brown_crew_sweatshirt | 圓領 | sweatshirt | 一般 |  | 1.21 | 1.04 | crew neck | 圓＋－ | long sleeves, loose sleeves, straight sleeves | 0.80 | 有 | CuffBand |
| i03_green_crew_tee | 圓領 | T-shirt | 一般 |  | 1.27 | 1.05 | crew neck | 圓＋－ | short sleeves, normal sleeves | 0.29 |  | － |
| i04_green_ribbed_long_sleeve_tee | 圓領 | sweatshirt | 一般 |  | 1.24 | 1.05 | crew neck | 圓＋－ | long sleeves, loose sleeves, straight sleeves | 0.90 | 有 | CuffBand |
| i05_maroon_rib_baby_tee | 圓領 | T-shirt | 一般 |  | 1.22 | 1.04 | crew neck | 圓＋－ | short sleeves, tight sleeves, straight sleeves | 0.23 |  | － |
| i06_navy_crew_sweatshirt | 圓領 | sweatshirt | 一般 |  | 1.20 | 1.04 | crew neck | 圓＋－ | long sleeves, loose sleeves, straight sleeves | 0.89 | 有 | CuffBand |
| i07_navy_crew_tee | 圓領 | T-shirt | 一般 |  | 1.27 | 1.05 | crew neck | 圓＋－ | short sleeves, normal sleeves | 0.30 |  | － |
| i08_olive_crew_long_sleeve_tee | 圓領 | sweatshirt | 一般 |  | 1.22 | 1.05 | crew neck | 圓＋－ | long sleeves, loose sleeves, straight sleeves | 0.89 | 有 | CuffBand |
| i09_pink_crew_tee | 圓領 | T-shirt | 一般 |  | 1.24 | 1.05 | crew neck | 圓＋－ | short sleeves, normal sleeves | 0.23 |  | － |
| i10_red_crew_long_sleeve_tee | 圓領 | long-sleeve shirt | 一般 |  | 1.23 | 1.04 | crew neck | 圓＋－ | long sleeves, normal sleeves, straight sleeves | 0.86 |  | － |
| i11_green_scoop_tee | 大圓領 | T-shirt | 一般 |  | 1.19 | 1.05 | crew neck | 圓＋－ | short sleeves, normal sleeves | 0.26 |  | － |
| i12_navy_scoop_long_sleeve_tee | 大圓領 | long-sleeve shirt | 一般 |  | 1.23 | 1.05 | crew neck | 圓＋－ | long sleeves, normal sleeves, straight sleeves | 0.97 |  | － |
| i13_purple_scoop_tee | 大圓領 | T-shirt | 一般 |  | 1.21 | 1.05 | V-neck | V＋－ | short sleeves, normal sleeves | 0.26 |  | － |
| i14_red_scoop_tee | 大圓領 | T-shirt | 一般 |  | 1.21 | 1.04 | crew neck | 圓(貝茲)＋－ | short sleeves, tight sleeves, straight sleeves | 0.18 |  | － |
| i15_royal_scoop_dolman_tee | 大圓領 | T-shirt | 一般 |  | 1.21 | 1.04 | crew neck | 圓＋－ | short sleeves, normal sleeves | 0.23 |  | － |
| i16_blue_deep_vneck_tee | V 領 | T-shirt（連身） | 合身 |  | 1.26 | 1.05 | V-neck | V＋－ | short sleeves, tight sleeves, straight sleeves | 0.36 |  | － |
| i17_charcoal_vneck_tee | V 領 | T-shirt（連身） | 合身 |  | 1.23 | 1.05 | V-neck | V＋－ | short sleeves, tight sleeves, straight sleeves | 0.42 |  | － |
| i18_green_vneck_tee | V 領 | T-shirt | 一般 |  | 1.20 | 1.05 | V-neck | V＋－ | short sleeves, tight sleeves, straight sleeves | 0.23 |  | － |
| i19_grey_vneck_sweater | V 領 | sweater | 一般 |  | 1.23 | 1.05 | V-neck | V＋－ | long sleeves, normal sleeves, straight sleeves | 0.90 | 有 | CuffBand |
| i20_navy_vneck_tee | V 領 | T-shirt | 一般 |  | 1.19 | 1.05 | V-neck | V＋－ | short sleeves, normal sleeves | 0.27 |  | － |
| i21_red_deep_vneck_tee | V 領 | T-shirt（連身） | 一般 |  | 1.21 | 1.05 | V-neck | V＋－ | short sleeves, tight sleeves, straight sleeves | 0.41 |  | － |
| i22_red_vneck_tee | V 領 | T-shirt | 合身 |  | 1.26 | 1.05 | V-neck | V＋－ | short sleeves, normal sleeves | 0.24 |  | － |
| i23_wine_vneck_sweater | V 領 | sweater | 一般 |  | 1.22 | 1.04 | V-neck | V＋－ | long sleeves, normal sleeves, straight sleeves | 0.91 | 有 | CuffBand |
| i24_black_square_neck_long_sleeve_top | 方領 | sweater | 一般 |  | 1.22 | 1.05 | crew neck | 圓＋－ | long sleeves, normal sleeves, straight sleeves | 0.95 | 有 | CuffBand |
| i25_burgundy_square_neck_tank | 方領 | tank top（連身） | 合身 |  | 1.29 | 1.06 | scoop neck | 圓＋－ | sleeveless | 無袖 |  | － |
| i26_grey_square_neck_tank | 方領 | tank top（連身） | 合身 |  | 1.26 | 1.06 | scoop neck | 圓＋－ | sleeveless | 無袖 |  | － |
| i27_navy_square_neck_rib_tee | 方領 | T-shirt | 一般 |  | 1.18 | 1.05 | crew neck | 圓＋－ | short sleeves, normal sleeves | 0.22 |  | － |
| i28_blue_boat_neck_top | 船領 | top garment | 一般 |  | 1.26 | 1.05 | crew neck | 圓＋－ | long sleeves, loose sleeves, straight sleeves | 0.97 |  | － |
| i29_green_boat_neck_sweater | 船領 | top | 一般 |  | 1.21 | 1.04 | boat neck | 圓(曲)＋－ | long sleeves, loose sleeves, straight sleeves | 0.97 |  | － |
| i30_red_boat_neck_top | 船領 | long-sleeve shirt | 一般 |  | 1.23 | 1.04 | crew neck | 圓＋－ | long sleeves, normal sleeves, straight sleeves | 0.91 |  | － |
| i31_teal_boat_neck_blouse | 船領 | dress（連身） | 合身 |  | 1.21 | 1.05 | crew neck | 圓＋－ | short sleeves, normal sleeves | 0.33 |  | － |
| i32_green_turtleneck_sweater | 高領／半高領 | sweater | 一般 |  | 1.24 | 1.05 | turtle neck | 圓＋高領 | long sleeves, loose sleeves, straight sleeves | 0.91 | 有 | CuffBand |
| i33_navy_mock_neck_tee | 高領／半高領 | long-sleeve shirt | 一般 |  | 1.20 | 1.05 | crew neck | 圓＋－ | long sleeves, normal sleeves, straight sleeves | 0.96 | 有 | CuffBand |
| i34_navy_turtleneck_sweater | 高領／半高領 | turtleneck sweater | 一般 |  | 1.20 | 1.05 | turtle neck | 圓＋高領 | long sleeves, loose sleeves, straight sleeves | 0.88 | 有 | CuffBand |
| i35_olive_mock_neck_tee | 高領／半高領 | T-shirt | 一般 |  | 1.23 | 1.05 | crew neck | 圓＋－ | short sleeves, normal sleeves | 0.16 |  | － |
| i36_red_turtleneck_sweater | 高領／半高領 | turtleneck sweater | 合身 |  | 1.20 | 1.05 | turtle neck | 圓＋高領 | long sleeves, loose sleeves, straight sleeves | 0.85 | 有 | CuffBand |
| i37_grey_henley_tee | 亨利領 | polo shirt | 一般 |  | 1.19 | 1.05 | polo collar | 圓＋翻領 | short sleeves, normal sleeves | 0.32 | 有 | － |
| i38_indigo_henley_tee | 亨利領 | T-shirt | 一般 |  | 1.21 | 1.04 | crew neck | 圓(曲)＋－ | short sleeves, normal sleeves | 0.20 |  | － |
| i39_red_henley_tee | 亨利領 | long-sleeve shirt | 一般 |  | 1.22 | 1.04 | crew neck | 圓＋－ | long sleeves, normal sleeves, straight sleeves | 0.96 | 有 | CuffBand |
| i40_charcoal_polo | Polo 翻領 | polo shirt | 一般 |  | 1.16 | 1.05 | polo collar | 圓(曲)＋翻領 | short sleeves, normal sleeves | 0.26 | 有 | － |
| i41_green_polo | Polo 翻領 | polo shirt | 一般 |  | 1.17 | 1.05 | polo collar | 圓(曲)＋翻領 | short sleeves, normal sleeves | 0.25 | 有 | － |
| i42_red_polo | Polo 翻領 | polo shirt | 一般 |  | 1.17 | 1.05 | polo collar | 圓(曲)＋翻領 | short sleeves, normal sleeves | 0.26 | 有 | － |
| i43_burgundy_dress_shirt | 襯衫領 | shirt | 一般 |  | 1.26 | 1.05 | point collar | 圓＋翻領 | long sleeves, normal sleeves, straight sleeves | 0.91 |  | － |
| i44_green_button_down_shirt | 襯衫領 | shirt | 一般 |  | 1.24 | 1.05 | spread collar | V＋翻領 | long sleeves, normal sleeves, straight sleeves | 0.86 | 有 | CuffBand |
| i45_grey_oxford_shirt | 襯衫領 | shirt | 一般 |  | 1.22 | 1.05 | point collar | 圓＋翻領 | long sleeves, normal sleeves, straight sleeves | 0.87 |  | － |
| i46_blue_hoodie | 帽 T | hoodie | 一般 |  | 1.18 | 1.04 | collarless | 圓＋帽子 | long sleeves, loose sleeves, straight sleeves | 0.84 | 有 | CuffBand |
| i47_green_hoodie | 帽 T | hoodie | 一般 |  | 1.18 | 1.04 | collarless | 圓＋帽子 | long sleeves, loose sleeves, straight sleeves | 0.83 | 有 | CuffBand |
| i48_red_hoodie | 帽 T | hoodie | 一般 |  | 1.22 | 1.05 | collarless | 圓＋帽子 | long sleeves, loose sleeves, straight sleeves | 0.81 | 有 | CuffBand |
| i49_gray_off_shoulder_top | 其他 | coat | 一般 | 開 | 1.27 | 1.05 | collarless | V＋－ | long sleeves, normal sleeves, straight sleeves | 0.87 |  | － |
| i50_red_sweetheart_neck_top | 其他 | T-shirt | 一般 |  | 1.22 | 1.04 | crew neck | 圓＋－ | short sleeves, tight sleeves, straight sleeves | 0.28 |  | － |
