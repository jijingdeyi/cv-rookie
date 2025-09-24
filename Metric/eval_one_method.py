import numpy as np
from PIL import Image
from Metric.metric import *
from natsort import natsorted
from tqdm import tqdm
import os
import warnings
from openpyxl import Workbook, load_workbook
from openpyxl.utils import get_column_letter

warnings.filterwarnings("ignore")


def write_excel(
    excel_name="metric.xlsx", worksheet_name="VIF", column_index=0, data=None
):
    """将 data 写入指定工作表的指定列（0-based）；自动移除默认空白 Sheet。"""
    if data is None:
        data = []

    try:
        workbook = load_workbook(excel_name)
        # 如果已有文件里仍保留默认空白表且还有其他表，顺手删掉
        if "Sheet" in workbook.sheetnames and len(workbook.sheetnames) > 1:
            workbook.remove(workbook["Sheet"])
    except FileNotFoundError:
        # 新建文件默认有个 'Sheet'，先删掉
        workbook = Workbook()
        if workbook.active and workbook.active.title == "Sheet":
            workbook.remove(workbook.active)

    # 获取或创建目标工作表
    worksheet = (
        workbook[worksheet_name]
        if worksheet_name in workbook.sheetnames
        else workbook.create_sheet(title=worksheet_name)
    )

    # 在指定列写数据
    col_letter = get_column_letter(column_index + 1)
    for i, value in enumerate(data, start=1):
        worksheet[f"{col_letter}{i}"].value = value

    # 再保险：若还有默认 'Sheet' 且存在其他表，移除之
    if (
        "Sheet" in workbook.sheetnames
        and worksheet_name != "Sheet"
        and len(workbook.sheetnames) > 1
    ):
        workbook.remove(workbook["Sheet"])

    workbook.save(excel_name)


def evaluation_one(ir_name, vi_name, f_name):
    f_img = Image.open(f_name).convert("L")
    ir_img = Image.open(ir_name).convert("L")
    vi_img = Image.open(vi_name).convert("L")

    f_img_int = np.array(f_img).astype(np.int32)
    f_img_double = np.array(f_img).astype(np.float32)

    ir_img_int = np.array(ir_img).astype(np.int32)
    ir_img_double = np.array(ir_img).astype(np.float32)

    vi_img_int = np.array(vi_img).astype(np.int32)
    vi_img_double = np.array(vi_img).astype(np.float32)

    EN = EN_function(f_img_int)
    MI = MI_function(ir_img_int, vi_img_int, f_img_int, gray_level=256)

    SF = SF_function(f_img_double)
    SD = SD_function(f_img_double)
    AG = AG_function(f_img_double)
    PSNR = PSNR_function(ir_img_double, vi_img_double, f_img_double)
    MSE = MSE_function(ir_img_double, vi_img_double, f_img_double)
    VIF = VIF_function(ir_img_double, vi_img_double, f_img_double)
    CC = CC_function(ir_img_double, vi_img_double, f_img_double)
    SCD = SCD_function(ir_img_double, vi_img_double, f_img_double)
    Qabf = Qabf_function(ir_img_double, vi_img_double, f_img_double)
    Nabf = Nabf_function(ir_img_double, vi_img_double, f_img_double)
    SSIM = SSIM_function(ir_img_double, vi_img_double, f_img_double)
    MS_SSIM = MS_SSIM_function(ir_img_double, vi_img_double, f_img_double)
    return EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM


if __name__ == "__main__":
    with_mean = True

    # --- 路径与配置 ---
    dataset_name = "msrs"
    Method = "ferfusion"  # 仅用于写入列首和文件名
    ir_dir = os.path.join("/data/ykx/MSRS_test", "ir")
    vi_dir = os.path.join("/data/ykx/MSRS_test", "vi")
    f_dir = os.path.join("/home/ykx/ReCoNet/result", "ori")  # 融合结果所在目录
    save_dir = os.path.join("/home/ykx/ReCoNet/result/metric", "ori")
    os.makedirs(save_dir, exist_ok=True)
    metric_save_name = os.path.join(save_dir, f"metric_{dataset_name}_{Method}.xlsx")

    # --- 计算指标 ---
    EN_list, MI_list, SF_list, AG_list, SD_list = [], [], [], [], []
    CC_list, SCD_list, VIF_list = [], [], []
    MSE_list, PSNR_list = [], []
    Qabf_list, Nabf_list = [], []
    SSIM_list, MS_SSIM_list = [], []
    filename_list = [""]  # 第一格空着，对齐“方法名”列

    filelist = natsorted(os.listdir(ir_dir))
    for item in tqdm(filelist, desc=f"Evaluating {Method}"):
        ir_name = os.path.join(ir_dir, item)
        vi_name = os.path.join(vi_dir, item)
        f_name = os.path.join(f_dir, item)
        # 如果融合结果缺失，跳过并给个占位（也可选择 raise）
        if not os.path.exists(f_name):
            # 你也可以选择 continue + 记录缺失文件
            raise FileNotFoundError(f"Fused image not found: {f_name}")

        EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM = (
            evaluation_one(ir_name, vi_name, f_name)
        )
        EN_list.append(EN)
        MI_list.append(MI)
        SF_list.append(SF)
        AG_list.append(AG)
        SD_list.append(SD)
        CC_list.append(CC)
        SCD_list.append(SCD)
        VIF_list.append(VIF)
        MSE_list.append(MSE)
        PSNR_list.append(PSNR)
        Qabf_list.append(Qabf)
        Nabf_list.append(Nabf)
        SSIM_list.append(SSIM)
        MS_SSIM_list.append(MS_SSIM)
        filename_list.append(item)

    if with_mean:
        # 用“原始样本”计算统计量，避免被 append 后的统计量污染
        EN_raw, MI_raw, SF_raw, AG_raw, SD_raw = (
            EN_list[:],
            MI_list[:],
            SF_list[:],
            AG_list[:],
            SD_list[:],
        )
        CC_raw, SCD_raw, VIF_raw = CC_list[:], SCD_list[:], VIF_list[:]
        MSE_raw, PSNR_raw = MSE_list[:], PSNR_list[:]
        Qabf_raw, Nabf_raw = Qabf_list[:], Nabf_list[:]
        SSIM_raw, MS_SSIM_raw = SSIM_list[:], MS_SSIM_list[:]

        # 均值
        EN_list.append(np.mean(EN_raw))
        MI_list.append(np.mean(MI_raw))
        SF_list.append(np.mean(SF_raw))
        AG_list.append(np.mean(AG_raw))
        SD_list.append(np.mean(SD_raw))
        CC_list.append(np.mean(CC_raw))
        SCD_list.append(np.mean(SCD_raw))
        VIF_list.append(np.mean(VIF_raw))
        MSE_list.append(np.mean(MSE_raw))
        PSNR_list.append(np.mean(PSNR_raw))
        Qabf_list.append(np.mean(Qabf_raw))
        Nabf_list.append(np.mean(Nabf_raw))
        SSIM_list.append(np.mean(SSIM_raw))
        MS_SSIM_list.append(np.mean(MS_SSIM_raw))
        filename_list.append("mean")

        # 标准差
        EN_list.append(np.std(EN_raw))
        MI_list.append(np.std(MI_raw))
        SF_list.append(np.std(SF_raw))
        AG_list.append(np.std(AG_raw))
        SD_list.append(np.std(SD_raw))
        CC_list.append(np.std(CC_raw))
        SCD_list.append(np.std(SCD_raw))
        VIF_list.append(np.std(VIF_raw))
        MSE_list.append(np.std(MSE_raw))
        PSNR_list.append(np.std(PSNR_raw))
        Qabf_list.append(np.std(Qabf_raw))
        Nabf_list.append(np.std(Nabf_raw))
        SSIM_list.append(np.std(SSIM_raw))
        MS_SSIM_list.append(np.std(MS_SSIM_raw))
        filename_list.append("std")

    # 保留三位小数
    round3 = lambda lst: [round(x, 3) for x in lst]
    EN_list = round3(EN_list)
    MI_list = round3(MI_list)
    SF_list = round3(SF_list)
    AG_list = round3(AG_list)
    SD_list = round3(SD_list)
    CC_list = round3(CC_list)
    SCD_list = round3(SCD_list)
    VIF_list = round3(VIF_list)
    MSE_list = round3(MSE_list)
    PSNR_list = round3(PSNR_list)
    Qabf_list = round3(Qabf_list)
    Nabf_list = round3(Nabf_list)
    SSIM_list = round3(SSIM_list)
    MS_SSIM_list = round3(MS_SSIM_list)

    # 每列顶部插入方法名
    for L in (
        EN_list,
        MI_list,
        SF_list,
        AG_list,
        SD_list,
        CC_list,
        SCD_list,
        VIF_list,
        MSE_list,
        PSNR_list,
        Qabf_list,
        Nabf_list,
        SSIM_list,
        MS_SSIM_list,
    ):
        L.insert(0, Method)

    # 逐表写入
    write_excel(metric_save_name, "EN", 0, filename_list)
    write_excel(metric_save_name, "MI", 0, filename_list)
    write_excel(metric_save_name, "SF", 0, filename_list)
    write_excel(metric_save_name, "AG", 0, filename_list)
    write_excel(metric_save_name, "SD", 0, filename_list)
    write_excel(metric_save_name, "CC", 0, filename_list)
    write_excel(metric_save_name, "SCD", 0, filename_list)
    write_excel(metric_save_name, "VIF", 0, filename_list)
    write_excel(metric_save_name, "MSE", 0, filename_list)
    write_excel(metric_save_name, "PSNR", 0, filename_list)
    write_excel(metric_save_name, "Qabf", 0, filename_list)
    write_excel(metric_save_name, "Nabf", 0, filename_list)
    write_excel(metric_save_name, "SSIM", 0, filename_list)
    write_excel(metric_save_name, "MS_SSIM", 0, filename_list)

    write_excel(metric_save_name, "EN", 1, EN_list)
    write_excel(metric_save_name, "MI", 1, MI_list)
    write_excel(metric_save_name, "SF", 1, SF_list)
    write_excel(metric_save_name, "AG", 1, AG_list)
    write_excel(metric_save_name, "SD", 1, SD_list)
    write_excel(metric_save_name, "CC", 1, CC_list)
    write_excel(metric_save_name, "SCD", 1, SCD_list)
    write_excel(metric_save_name, "VIF", 1, VIF_list)
    write_excel(metric_save_name, "MSE", 1, MSE_list)
    write_excel(metric_save_name, "PSNR", 1, PSNR_list)
    write_excel(metric_save_name, "Qabf", 1, Qabf_list)
    write_excel(metric_save_name, "Nabf", 1, Nabf_list)
    write_excel(metric_save_name, "SSIM", 1, SSIM_list)
    write_excel(metric_save_name, "MS_SSIM", 1, MS_SSIM_list)
