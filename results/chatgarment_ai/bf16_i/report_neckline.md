# 第三批領口比對（bf16，50 件）

由 `tools/chatgarment/experiments/neck_report.py` 產生。正確答案＝`labels.json` 的組別（照商品頁寫的領型）。

- **描述**：AI 第一步文字描述的 collar／hood 有沒有寫出正確領型
- **原始版型**：AI 第二步的 GarmentCode 參數（前領形狀＋領子配件）
- **修正後**：`to_design.py` 以描述為準修正後
- 船領、心形領、一字領 GarmentCode 沒有對應形狀，只看領子配件；Polo、襯衫、帽 T 主要看領子配件
- **前領深**：AI 原本 → `neckline.py` 校正後（cm，平均男性身材）

## 各組統計

| 組 | 領型 | 件數 | 描述對 | 原始版型對 | 修正後對 |
|---|---|---|---|---|---|
| A | 圓領 | 10 | 10 | 7 | 10 |
| B | 大圓領 | 5 | 0 | 3 | 4 |
| C | V 領 | 8 | 8 | 0 | 8 |
| D | 方領 | 4 | 0 | 0 | 0 |
| E | 船領 | 4 | 1 | 2 | 4 |
| F | 高領／半高領 | 5 | 3 | 4 | 3 |
| G | 亨利領 | 3 | 0 | 2 | 2 |
| H | Polo 翻領 | 3 | 3 | 0 | 3 |
| I | 襯衫領 | 3 | 3 | 1 | 3 |
| J | 帽 T | 3 | 3 | 3 | 3 |
| K | 其他 | 2 | 0 | 1 | 2 |
| | **合計** | **50** | **31** | **23** | **42** |

## 每件

| 衣服 | 正確答案 | AI 描述 | | 原始版型 | | 修正後 | | 備註 | 前領深 cm |
|---|---|---|---|---|---|---|---|---|---|
| i01_blue_crew_tee | A 圓領 | crew neck | ✔ | 方＋高領 | ✘ | 圓＋無 | ✔ ✔ 修好 |  | 13.4 → 7.9 |
| i02_brown_crew_sweatshirt | A 圓領 | crew neck | ✔ | 圓＋無 | ✔ | 圓＋無 | ✔  |  | 13.2 → 7.9 |
| i03_green_crew_tee | A 圓領 | crew neck | ✔ | 圓＋無 | ✔ | 圓＋無 | ✔  |  | 14.3 → 9.1 |
| i04_green_ribbed_long_sleeve_tee | A 圓領 | crew neck | ✔ | 圓＋高領 | ✘ | 圓＋無 | ✔ ✔ 修好 |  | 14.1 → 8.8 |
| i05_maroon_rib_baby_tee | A 圓領 | crew neck | ✔ | 圓＋無 | ✔ | 圓＋無 | ✔  |  | 13.7 → 8.3 |
| i06_navy_crew_sweatshirt | A 圓領 | crew neck | ✔ | 圓＋無 | ✔ | 圓＋無 | ✔  |  | 13.3 → 7.9 |
| i07_navy_crew_tee | A 圓領 | crew neck | ✔ | 圓＋無 | ✔ | 圓＋無 | ✔  |  | 14.8 → 9.6 |
| i08_olive_crew_long_sleeve_tee | A 圓領 | crew neck | ✔ | 圓＋無 | ✔ | 圓＋無 | ✔  |  | 13.7 → 8.3 |
| i09_pink_crew_tee | A 圓領 | crew neck | ✔ | 圓＋高領 | ✘ | 圓＋無 | ✔ ✔ 修好 |  | 量不到 |
| i10_red_crew_long_sleeve_tee | A 圓領 | crew neck | ✔ | 圓＋無 | ✔ | 圓＋無 | ✔  |  | 13.6 → 8.3 |
| i11_green_scoop_tee | B 大圓領 | crew neck | ✘ | 圓＋高領 | ✘ | 圓＋無 | ✔ ✔ 修好 |  | 13.6 → 8.2 |
| i12_navy_scoop_long_sleeve_tee | B 大圓領 | crew neck | ✘ | 圓＋無 | ✔ | 圓＋無 | ✔  |  | 14.4 → 9.1 |
| i13_purple_scoop_tee | B 大圓領 | V-neck | ✘ | 圓＋高領 | ✘ | V＋無 | ✘ 改了 |  | 13.6 → 13.5 |
| i14_red_scoop_tee | B 大圓領 | crew neck | ✘ | 圓(貝茲)＋無 | ✔ | 圓(貝茲)＋無 | ✔  |  | 13.4 → 7.9 |
| i15_royal_scoop_dolman_tee | B 大圓領 | crew neck | ✘ | 圓＋無 | ✔ | 圓＋無 | ✔  |  | 13.6 → 8.3 |
| i16_blue_deep_vneck_tee | C V 領 | V-neck | ✔ | V＋高領 | ✘ | V＋無 | ✔ ✔ 修好 | 連身 | 12.9 → 15.5 |
| i17_charcoal_vneck_tee | C V 領 | V-neck | ✔ | 梯形＋高領 | ✘ | V＋無 | ✔ ✔ 修好 | 連身 | 13.5 → 16.0 |
| i18_green_vneck_tee | C V 領 | V-neck | ✔ | 圓＋無 | ✘ | V＋無 | ✔ ✔ 修好 |  | 13.4 → 13.2 |
| i19_grey_vneck_sweater | C V 領 | V-neck | ✔ | 圓＋高領 | ✘ | V＋無 | ✔ ✔ 修好 |  | 14.1 → 14.0 |
| i20_navy_vneck_tee | C V 領 | V-neck | ✔ | V＋翻領 | ✘ | V＋無 | ✔ ✔ 修好 |  | 14.5 → 14.3 |
| i21_red_deep_vneck_tee | C V 領 | V-neck | ✔ | 梯形＋無 | ✘ | V＋無 | ✔ ✔ 修好 | 連身 | 量不到 |
| i22_red_vneck_tee | C V 領 | V-neck | ✔ | 圓(貝茲)＋無 | ✘ | V＋無 | ✔ ✔ 修好 |  | 13.0 → 15.5 |
| i23_wine_vneck_sweater | C V 領 | V-neck | ✔ | 圓＋無 | ✘ | V＋無 | ✔ ✔ 修好 |  | 14.3 → 14.2 |
| i24_black_square_neck_long_sleeve_top | D 方領 | crew neck | ✘ | 圓＋無 | ✘ | 圓＋無 | ✘  |  | 14.3 → 9.0 |
| i25_burgundy_square_neck_tank | D 方領 | scoop neck | ✘ | 圓＋無 | ✘ | 圓＋無 | ✘  | 連身 | 14.5 → 12.1 |
| i26_grey_square_neck_tank | D 方領 | scoop neck | ✘ | 圓＋無 | ✘ | 圓＋無 | ✘  | 連身 | 14.6 → 12.2 |
| i27_navy_square_neck_rib_tee | D 方領 | crew neck | ✘ | 圓＋高領 | ✘ | 圓＋無 | ✘ 改了 |  | 13.6 → 8.2 |
| i28_blue_boat_neck_top | E 船領 | crew neck | ✘ | 方＋翻領 | ✘ | 圓＋無 | ✔ ✔ 修好 |  | 15.6 → 10.4 |
| i29_green_boat_neck_sweater | E 船領 | boat neck | ✔ | 圓(曲)＋無 | ✔ | 圓(曲)＋無 | ✔  |  | 16.1 → 11.0 |
| i30_red_boat_neck_top | E 船領 | crew neck | ✘ | 圓＋無 | ✔ | 圓＋無 | ✔  |  | 13.8 → 8.5 |
| i31_teal_boat_neck_blouse | E 船領 | crew neck | ✘ | 圓＋翻領 | ✘ | 圓＋無 | ✔ ✔ 修好 | 連身 | 12.8 → 10.5 |
| i32_green_turtleneck_sweater | F 高領／半高領 | turtle neck | ✔ | 圓＋高領 | ✔ | 圓＋高領 | ✔  |  | 15.1 → 9.9 |
| i33_navy_mock_neck_tee | F 高領／半高領 | crew neck | ✘ | 圓＋高領 | ✔ | 圓＋無 | ✘ ✘ 改錯 |  | 14.9 → 9.6 |
| i34_navy_turtleneck_sweater | F 高領／半高領 | turtle neck | ✔ | 圓＋高領 | ✔ | 圓＋高領 | ✔  |  | 14.3 → 8.9 |
| i35_olive_mock_neck_tee | F 高領／半高領 | crew neck | ✘ | 方＋高領 | ✘ | 圓＋無 | ✘ 改了 |  | 13.4 → 8.0 |
| i36_red_turtleneck_sweater | F 高領／半高領 | turtle neck | ✔ | 圓＋高領 | ✔ | 圓＋高領 | ✔  |  | 14.0 → 11.5 |
| i37_grey_henley_tee | G 亨利領 | polo collar | ✘ | 圓＋翻領 | ✘ | 圓＋翻領 | ✘  |  | 13.3 → 8.0 |
| i38_indigo_henley_tee | G 亨利領 | crew neck | ✘ | 圓(曲)＋無 | ✔ | 圓(曲)＋無 | ✔  |  | 13.5 → 8.0 |
| i39_red_henley_tee | G 亨利領 | crew neck | ✘ | 圓＋無 | ✔ | 圓＋無 | ✔  |  | 14.4 → 9.1 |
| i40_charcoal_polo | H Polo 翻領 | polo collar | ✔ | 圓(曲)＋無 | ✘ | 圓(曲)＋翻領 | ✔ ✔ 修好 |  | 13.4 → 8.0 |
| i41_green_polo | H Polo 翻領 | polo collar | ✔ | 圓(曲)＋無 | ✘ | 圓(曲)＋翻領 | ✔ ✔ 修好 |  | 13.4 → 7.9 |
| i42_red_polo | H Polo 翻領 | polo collar | ✔ | 圓(曲)＋無 | ✘ | 圓(曲)＋翻領 | ✔ ✔ 修好 |  | 13.5 → 8.1 |
| i43_burgundy_dress_shirt | I 襯衫領 | point collar | ✔ | 圓＋高領 | ✘ | 圓＋翻領 | ✔ ✔ 修好 |  | 15.2 → 10.0 |
| i44_green_button_down_shirt | I 襯衫領 | spread collar | ✔ | V＋無 | ✘ | V＋翻領 | ✔ ✔ 修好 |  | 15.2 → 15.0 |
| i45_grey_oxford_shirt | I 襯衫領 | point collar | ✔ | 圓＋翻領 | ✔ | 圓＋翻領 | ✔  |  | 14.9 → 9.7 |
| i46_blue_hoodie | J 帽 T | collarless, normal hood | ✔ | 圓＋帽子 | ✔ | 圓＋帽子 | ✔  |  | 14.7 → 9.5 |
| i47_green_hoodie | J 帽 T | collarless, normal hood | ✔ | 圓＋帽子 | ✔ | 圓＋帽子 | ✔  |  | 14.8 → 9.5 |
| i48_red_hoodie | J 帽 T | collarless, normal hood | ✔ | 圓＋帽子 | ✔ | 圓＋帽子 | ✔  |  | 14.3 → 9.0 |
| i49_gray_off_shoulder_top | K 其他 | collarless | ✘ | V＋無 | ✔ | V＋無 | ✔  |  | 16.1 → 15.9 |
| i50_red_sweetheart_neck_top | K 其他 | crew neck | ✘ | 圓＋高領 | ✘ | 圓＋無 | ✔ ✔ 修好 |  | 13.4 → 8.0 |
