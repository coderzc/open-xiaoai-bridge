"""
Doubao (ByteDance Volcano Engine) TTS Service
"""

import base64
import json
import io
import requests
import numpy as np
import soundfile as sf
from typing import Optional, AsyncGenerator
from scipy import signal


class DoubaoTTS:
    """豆包语音合成服务 (字节跳动火山引擎)"""

    DEFAULT_URL = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"

    # 官方音色列表 - 来自 https://www.volcengine.com/docs/6561/1257544

    # 豆包语音合成模型2.0 音色 (seed-tts-2.0)
    # 支持情感变化、指令遵循、ASMR等能力
    VOICES_2_0 = {
        # ===== 通用场景 =====
        "zh_female_vv_uranus_bigtts": "Vivi 2.0",
        "zh_female_xiaohe_uranus_bigtts": "小何 2.0",
        "zh_male_m191_uranus_bigtts": "云舟 2.0",
        "zh_male_taocheng_uranus_bigtts": "小天 2.0",

        # ===== 有声阅读 =====
        "zh_female_xueayi_saturn_bigtts": "儿童绘本",

        # ===== 视频配音 =====
        "zh_male_dayi_saturn_bigtts": "大壹",
        "zh_female_mizai_saturn_bigtts": "黑猫侦探社咪",
        "zh_female_jitangnv_saturn_bigtts": "鸡汤女",
        "zh_female_meilinvyou_saturn_bigtts": "魅力女友",
        "zh_female_santongyongns_saturn_bigtts": "流畅女声",
        "zh_male_ruyayichen_saturn_bigtts": "儒雅逸辰",

        # ===== 角色扮演 =====
        "saturn_zh_female_keainvsheng_tob": "可爱女生",
        "saturn_zh_female_tiaopigongzhu_tob": "调皮公主",
        "saturn_zh_male_shuanglangshaonian_tob": "爽朗少年",
        "saturn_zh_male_tiancaitongzhuo_tob": "天才同桌",
        "saturn_zh_female_cancan_tob": "知性灿灿",

        # ===== 客服场景 =====
        "saturn_zh_female_qingyingduoduo_cs_tob": "轻盈朵朵 2.0",
        "saturn_zh_female_wenwanshanshan_cs_tob": "温婉珊珊 2.0",
        "saturn_zh_female_reqingaina_cs_tob": "热情艾娜 2.0",

        # ===== 多语种 (美式英语) =====
        "en_male_tim_uranus_bigtts": "Tim",
        "en_female_dacey_uranus_bigtts": "Dacey",
        "en_female_stokie_uranus_bigtts": "Stokie",
    }

    # 豆包语音合成模型1.0 音色 (seed-tts-1.0)
    VOICES_1_0 = {
        # ===== 多情感音色 (支持 emotion 参数) =====
        "zh_male_lengkugege_emo_v2_mars_bigtts": "冷酷哥哥(多情感)",
        "zh_female_tianxinxiaomei_emo_v2_mars_bigtts": "甜心小美(多情感)",
        "zh_female_gaolengyujie_emo_v2_mars_bigtts": "高冷御姐(多情感)",
        "zh_male_aojiaobazong_emo_v2_mars_bigtts": "傲娇霸总(多情感)",
        "zh_male_guangzhoudege_emo_mars_bigtts": "广州德哥(多情感/粤普)",
        "zh_male_jingqiangkanye_emo_mars_bigtts": "京腔侃爷(多情感/北京)",
        "zh_female_linjuayi_emo_v2_mars_bigtts": "邻居阿姨(多情感)",
        "zh_male_yourougongzi_emo_v2_mars_bigtts": "优柔公子(多情感)",
        "zh_male_ruyayichen_emo_v2_mars_bigtts": "儒雅男友(多情感)",
        "zh_male_junlangnanyou_emo_v2_mars_bigtts": "俊朗男友(多情感)",
        "zh_male_beijingxiaoye_emo_v2_mars_bigtts": "北京小爷(多情感)",
        "zh_female_roumeinvyou_emo_v2_mars_bigtts": "柔美女友(多情感)",
        "zh_male_yangguangqingnian_emo_v2_mars_bigtts": "阳光青年(多情感)",
        "zh_female_meilinvyou_emo_v2_mars_bigtts": "魅力女友(多情感)",
        "zh_female_shuangkuaisisi_emo_v2_mars_bigtts": "爽快思思(多情感)",
        "en_female_candice_emo_v2_mars_bigtts": "Candice(多情感/英)",
        "en_female_skye_emo_v2_mars_bigtts": "Serena(多情感/英)",
        "en_male_glen_emo_v2_mars_bigtts": "Glen(多情感/英)",
        "en_male_sylus_emo_v2_mars_bigtts": "Sylus(多情感/英)",
        "en_male_corey_emo_v2_mars_bigtts": "Corey(多情感/英)",
        "en_female_nadia_tips_emo_v2_mars_bigtts": "Nadia(多情感/英)",
        "zh_male_shenyeboke_emo_v2_mars_bigtts": "深夜播客(多情感)",
        "zh_male_zhoujielun_emo_v2_mars_bigtts": "双节棍小哥(多情感/台湾)",

        # ===== 通用场景 =====
        "ICL_zh_female_wenrounvshen_239eff5e8ffa_tob": "温柔女神",
        "zh_female_vv_mars_bigtts": "Vivi",
        "zh_female_qinqienvsheng_moon_bigtts": "亲切女声",
        "ICL_zh_male_shenmi_v1_tob": "机灵小伙",
        "ICL_zh_female_wuxi_tob": "元气甜妹",
        "ICL_zh_female_wenyinvsheng_v1_tob": "知心姐姐",
        "zh_male_qingyiyuxuan_mars_bigtts": "阳光阿辰",
        "zh_male_xudong_conversation_wvae_bigtts": "快乐小东",
        "ICL_zh_male_lengkugege_v1_tob": "冷酷哥哥",
        "ICL_zh_female_feicui_v1_tob": "纯澈女生",
        "ICL_zh_female_yuxin_v1_tob": "初恋女友",
        "ICL_zh_female_xnx_tob": "贴心闺蜜",
        "ICL_zh_female_yry_tob": "温柔白月光",
        "ICL_zh_male_BV705_streaming_cs_tob": "炀炀",
        "en_male_jason_conversation_wvae_bigtts": "开朗学长",
        "zh_female_sophie_conversation_wvae_bigtts": "魅力苏菲",
        "ICL_zh_female_yilin_tob": "贴心妹妹",
        "zh_female_tianmeitaozi_mars_bigtts": "甜美桃子",
        "zh_female_qingxinnvsheng_mars_bigtts": "清新女声",
        "zh_female_zhixingnvsheng_mars_bigtts": "知性女声",
        "zh_male_qingshuangnanda_mars_bigtts": "清爽男大",
        "zh_female_linjianvhai_moon_bigtts": "邻家女孩",
        "zh_male_yuanboxiaoshu_moon_bigtts": "渊博小叔",
        "zh_male_yangguangqingnian_moon_bigtts": "阳光青年",
        "zh_female_tianmeixiaoyuan_moon_bigtts": "甜美小源",
        "zh_female_qingchezizi_moon_bigtts": "清澈梓梓",
        "zh_male_jieshuoxiaoming_moon_bigtts": "解说小明",
        "zh_female_kailangjiejie_moon_bigtts": "开朗姐姐",
        "zh_male_linjiananhai_moon_bigtts": "邻家男孩",
        "zh_female_tianmeiyueyue_moon_bigtts": "甜美悦悦",
        "zh_female_xinlingjitang_moon_bigtts": "心灵鸡汤",
        "ICL_zh_female_zhixingwenwan_tob": "知性温婉",
        "ICL_zh_male_nuanxintitie_tob": "暖心体贴",
        "ICL_zh_male_kailangqingkuai_tob": "开朗轻快",
        "ICL_zh_male_huoposhuanglang_tob": "活泼爽朗",
        "ICL_zh_male_shuaizhenxiaohuo_tob": "率真小伙",
        "zh_male_wenrouxiaoge_mars_bigtts": "温柔小哥",
        "zh_female_cancan_mars_bigtts": "灿灿/Shiny",
        "zh_female_shuangkuaisisi_moon_bigtts": "爽快思思/Skye",
        "zh_male_wennuanahu_moon_bigtts": "温暖阿虎/Alvin",
        "zh_male_shaonianzixin_moon_bigtts": "少年梓辛/Brayan",
        "ICL_zh_female_wenrouwenya_tob": "温柔文雅",

        # ===== IP仿音 =====
        "zh_male_hupunan_mars_bigtts": "沪普男",
        "zh_male_lubanqihao_mars_bigtts": "鲁班七号",
        "zh_female_yangmi_mars_bigtts": "林潇",
        "zh_female_linzhiling_mars_bigtts": "玲玲姐姐",
        "zh_female_jiyejizi2_mars_bigtts": "春日部姐姐",
        "zh_male_tangseng_mars_bigtts": "唐僧",
        "zh_male_zhuangzhou_mars_bigtts": "庄周",
        "zh_male_zhubajie_mars_bigtts": "猪八戒",
        "zh_female_ganmaodianyin_mars_bigtts": "感冒电音姐姐",
        "zh_female_naying_mars_bigtts": "直率英子",
        "zh_female_leidian_mars_bigtts": "女雷神",

        # ===== 趣味口音 =====
        "zh_female_yueyunv_mars_bigtts": "粤语小溏",
        "zh_male_yuzhouzixuan_moon_bigtts": "豫州子轩(河南)",
        "zh_female_daimengchuanmei_moon_bigtts": "呆萌川妹(四川)",
        "zh_male_guangxiyuanzhou_moon_bigtts": "广西远舟",
        "zh_female_wanwanxiaohe_moon_bigtts": "湾湾小何(台湾)",
        "zh_female_wanqudashu_moon_bigtts": "湾区大叔(广东)",
        "zh_male_guozhoudege_moon_bigtts": "广州德哥(广东)",
        "zh_male_haoyuxiaoge_moon_bigtts": "浩宇小哥(青岛)",
        "zh_male_beijingxiaoye_moon_bigtts": "北京小爷(北京)",
        "zh_male_jingqiangkanye_moon_bigtts": "京腔侃爷/Harmony(北京)",
        "zh_female_meituojieer_moon_bigtts": "妹坨洁儿(长沙)",

        # ===== 角色扮演 =====
        "ICL_zh_female_chunzhenshaonv_e588402fb8ad_tob": "纯真少女",
        "ICL_zh_male_xiaonaigou_edf58cf28b8b_tob": "奶气小生",
        "ICL_zh_female_jinglingxiangdao_1beb294a9e3e_tob": "精灵向导",
        "ICL_zh_male_menyoupingxiaoge_ffed9fc2fee7_tob": "闷油瓶小哥",
        "ICL_zh_male_anrenqinzhu_cd62e63dcdab_tob": "黯刃秦主",
        "ICL_zh_male_badaozongcai_v1_tob": "霸道总裁",
        "ICL_zh_female_ganli_v1_tob": "妩媚可人",
        "ICL_zh_female_xiangliangya_v1_tob": "邪魅御姐",
        "ICL_zh_male_ms_tob": "嚣张小哥",
        "ICL_zh_male_you_tob": "油腻大叔",
        "ICL_zh_male_guaogongzi_v1_tob": "孤傲公子",
        "ICL_zh_male_huzi_v1_tob": "胡子叔叔",
        "ICL_zh_female_luoqing_v1_tob": "性感魅惑",
        "ICL_zh_male_bingruogongzi_tob": "病弱公子",
        "ICL_zh_female_bingjiao3_tob": "邪魅女王",
        "ICL_zh_male_aomanqingnian_tob": "傲慢青年",
        "ICL_zh_male_cujingnansheng_tob": "醋精男生",
        "ICL_zh_male_shuanglangshaonian_tob": "爽朗少年",
        "ICL_zh_male_sajiaonanyou_tob": "撒娇男友",
        "ICL_zh_male_wenrounanyou_tob": "温柔男友",
        "ICL_zh_male_wenshunshaonian_tob": "温顺少年",
        "ICL_zh_male_naigounanyou_tob": "粘人男友",
        "ICL_zh_male_sajiaonansheng_tob": "撒娇男生",
        "ICL_zh_male_huoponanyou_tob": "活泼男友",
        "ICL_zh_male_tianxinanyou_tob": "甜系男友",
        "ICL_zh_male_huoliqingnian_tob": "活力青年",
        "ICL_zh_male_kailangqingnian_tob": "开朗青年",
        "ICL_zh_male_lengmoxiongzhang_tob": "冷漠兄长",
        "ICL_zh_male_tiancaitongzhuo_tob": "天才同桌",
        "ICL_zh_male_pianpiangongzi_tob": "翩翩公子",
        "ICL_zh_male_mengdongqingnian_tob": "懵懂青年",
        "ICL_zh_male_lenglianxiongzhang_tob": "冷脸兄长",
        "ICL_zh_male_bingjiaoshaonian_tob": "病娇少年",
        "ICL_zh_male_bingjiaonanyou_tob": "病娇男友",
        "ICL_zh_male_bingruoshaonian_tob": "病弱少年",
        "ICL_zh_male_yiqishaonian_tob": "意气少年",
        "ICL_zh_male_ganjingshaonian_tob": "干净少年",
        "ICL_zh_male_lengmonanyou_tob": "冷漠男友",
        "ICL_zh_male_jingyingqingnian_tob": "精英青年",
        "ICL_zh_male_rexueshaonian_tob": "热血少年",
        "ICL_zh_male_qingshuangshaonian_tob": "清爽少年",
        "ICL_zh_male_zhongerqingnian_tob": "中二青年",
        "ICL_zh_male_lingyunqingnian_tob": "凌云青年",
        "ICL_zh_male_zifuqingnian_tob": "自负青年",
        "ICL_zh_male_bujiqingnian_tob": "不羁青年",
        "ICL_zh_male_ruyajunzi_tob": "儒雅君子",
        "ICL_zh_male_diyinchenyu_tob": "低音沉郁",
        "ICL_zh_male_lenglianxueba_tob": "冷脸学霸",
        "ICL_zh_male_ruyazongcai_tob": "儒雅总裁",
        "ICL_zh_male_shenchenzongcai_tob": "深沉总裁",
        "ICL_zh_male_xiaohouye_tob": "小侯爷",
        "ICL_zh_male_gugaogongzi_tob": "孤高公子",
        "ICL_zh_male_zhangjianjunzi_tob": "仗剑君子",
        "ICL_zh_male_wenrunxuezhe_tob": "温润学者",
        "ICL_zh_male_qinqieqingnian_tob": "亲切青年",
        "ICL_zh_male_wenrouxuezhang_tob": "温柔学长",
        "ICL_zh_male_gaolengzongcai_tob": "高冷总裁",
        "ICL_zh_male_lengjungaozhi_tob": "冷峻高智",
        "ICL_zh_male_chanruoshaoye_tob": "孱弱少爷",
        "ICL_zh_male_zixinqingnian_tob": "自信青年",
        "ICL_zh_male_qingseqingnian_tob": "青涩青年",
        "ICL_zh_male_xuebatongzhuo_tob": "学霸同桌",
        "ICL_zh_male_lengaozongcai_tob": "冷傲总裁",
        "ICL_zh_male_yuanqishaonian_tob": "元气少年",
        "ICL_zh_male_satuoqingnian_tob": "洒脱青年",
        "ICL_zh_male_zhishuaiqingnian_tob": "直率青年",
        "ICL_zh_male_siwenqingnian_tob": "斯文青年",
        "ICL_zh_male_junyigongzi_tob": "俊逸公子",
        "ICL_zh_male_zhangjianxiake_tob": "仗剑侠客",
        "ICL_zh_male_jijiaozhineng_tob": "机甲智能",
        "zh_male_naiqimengwa_mars_bigtts": "奶气萌娃",
        "zh_female_popo_mars_bigtts": "婆婆",
        "zh_female_gaolengyujie_moon_bigtts": "高冷御姐",
        "zh_male_aojiaobazong_moon_bigtts": "傲娇霸总",
        "zh_female_meilinvyou_moon_bigtts": "魅力女友",
        "zh_male_shenyeboke_moon_bigtts": "深夜播客",
        "zh_female_sajiaonvyou_moon_bigtts": "柔美女友",
        "zh_female_yuanqinvyou_moon_bigtts": "撒娇学妹",
        "ICL_zh_female_bingruoshaonv_tob": "病弱少女",
        "ICL_zh_female_huoponvhai_tob": "活泼女孩",
        "zh_male_dongfanghaoran_moon_bigtts": "东方浩然",
        "ICL_zh_male_lvchaxiaoge_tob": "绿茶小哥",
        "ICL_zh_female_jiaoruoluoli_tob": "娇喘萝莉",
        "ICL_zh_male_lengdanshuli_tob": "冷淡疏离",
        "ICL_zh_male_hanhoudunshi_tob": "憨厚敦实",
        "ICL_zh_female_huopodiaoman_tob": "活泼刁蛮",
        "ICL_zh_male_guzhibingjiao_tob": "固执病娇",
        "ICL_zh_male_sajiaonianren_tob": "撒娇粘人",
        "ICL_zh_female_aomanjiaosheng_tob": "傲慢娇声",
        "ICL_zh_male_xiaosasuixing_tob": "潇洒随性",
        "ICL_zh_male_guiyishenmi_tob": "诡异神秘",
        "ICL_zh_male_ruyacaijun_tob": "儒雅才俊",
        "ICL_zh_male_zhengzhiqingnian_tob": "正直青年",
        "ICL_zh_female_jiaohannvwang_tob": "娇憨女王",
        "ICL_zh_female_bingjiaomengmei_tob": "病娇萌妹",
        "ICL_zh_male_qingsenaigou_tob": "青涩小生",
        "ICL_zh_male_chunzhenxuedi_tob": "纯真学弟",
        "ICL_zh_male_youroubangzhu_tob": "优柔帮主",
        "ICL_zh_male_yourougongzi_tob": "优柔公子",
        "ICL_zh_female_tiaopigongzhu_tob": "调皮公主",
        "ICL_zh_male_tiexinnanyou_tob": "贴心男友",
        "ICL_zh_male_shaonianjiangjun_tob": "少年将军",
        "ICL_zh_male_bingjiaogege_tob": "病娇哥哥",
        "ICL_zh_male_xuebanantongzhuo_tob": "学霸男同桌",
        "ICL_zh_male_youmoshushu_tob": "幽默叔叔",
        "ICL_zh_female_jiaxiaozi_tob": "假小子",
        "ICL_zh_male_wenrounantongzhuo_tob": "温柔男同桌",
        "ICL_zh_male_youmodaye_tob": "幽默大爷",
        "ICL_zh_male_asmryexiu_tob": "枕边低语",
        "ICL_zh_male_shenmifashi_tob": "神秘法师",
        "zh_female_jiaochuan_mars_bigtts": "娇喘女声",
        "zh_male_livelybro_mars_bigtts": "开朗弟弟",
        "zh_female_flattery_mars_bigtts": "谄媚女声",
        "ICL_zh_male_lengjunshangsi_tob": "冷峻上司",
        "ICL_zh_male_xiaoge_v1_tob": "寡言小哥",
        "ICL_zh_male_renyuwangzi_v1_tob": "清朗温润",
        "ICL_zh_male_xiaosha_v1_tob": "潇洒随性",
        "ICL_zh_male_liyisheng_v1_tob": "清冷矜贵",
        "ICL_zh_male_qinglen_v1_tob": "沉稳优雅",
        "ICL_zh_male_chongqingzhanzhan_v1_tob": "清逸苏感",
        "ICL_zh_male_xingjiwangzi_v1_tob": "温柔内敛",
        "ICL_zh_male_sigeshiye_v1_tob": "低沉缱绻",
        "ICL_zh_male_lanyingcaohunshi_v1_tob": "蓝银草魂师",
        "ICL_zh_female_liumengdie_v1_tob": "清冷高雅",
        "ICL_zh_female_linxueying_v1_tob": "甜美娇俏",
        "ICL_zh_female_rouguhunshi_v1_tob": "柔骨魂师",
        "ICL_zh_female_tianmei_v1_tob": "甜美活泼",
        "ICL_zh_female_chengshu_v1_tob": "成熟温柔",
        "ICL_zh_female_xnx_v1_tob": "贴心闺蜜",
        "ICL_zh_female_yry_v1_tob": "温柔白月光",
        "zh_male_bv139_audiobook_ummv3_bigtts": "高冷沉稳",
        "ICL_zh_male_cujingnanyou_tob": "醋精男友",
        "ICL_zh_male_fengfashaonian_tob": "风发少年",
        "ICL_zh_male_cixingnansang_tob": "磁性男嗓",
        "ICL_zh_male_chengshuzongcai_tob": "成熟总裁",
        "ICL_zh_male_aojiaojingying_tob": "傲娇精英",
        "ICL_zh_male_aojiaogongzi_tob": "傲娇公子",
        "ICL_zh_male_badaoshaoye_tob": "霸道少爷",
        "ICL_zh_male_fuheigongzi_tob": "腹黑公子",
        "ICL_zh_female_nuanxinxuejie_tob": "暖心学姐",
        "ICL_zh_female_keainvsheng_tob": "可爱女生",
        "ICL_zh_female_chengshujiejie_tob": "成熟姐姐",
        "ICL_zh_female_bingjiaojiejie_tob": "病娇姐姐",
        "ICL_zh_female_wumeiyujie_tob": "妩媚御姐",
        "ICL_zh_female_aojiaonvyou_tob": "傲娇女友",
        "ICL_zh_female_tiexinnvyou_tob": "贴心女友",
        "ICL_zh_female_xingganyujie_tob": "性感御姐",
        "ICL_zh_male_bingjiaodidi_tob": "病娇弟弟",
        "ICL_zh_male_aomanshaoye_tob": "傲慢少爷",
        "ICL_zh_male_aiqilingren_tob": "傲气凌人",
        "ICL_zh_male_bingjiaobailian_tob": "病娇白莲",

        # ===== 教育场景 =====
        "zh_female_yingyujiaoyu_mars_bigtts": "Tina老师",

        # ===== 客服场景 =====
        "ICL_zh_female_lixingyuanzi_cs_tob": "理性圆子(客服)",
        "ICL_zh_female_qingtiantaotao_cs_tob": "清甜桃桃(客服)",
        "ICL_zh_female_qingxixiaoxue_cs_tob": "清晰小雪(客服)",
        "ICL_zh_female_qingtianmeimei_cs_tob": "清甜莓莓(客服)",
        "ICL_zh_female_kailangtingting_cs_tob": "开朗婷婷(客服)",
        "ICL_zh_male_qingxinmumu_cs_tob": "清新沐沐(客服)",
        "ICL_zh_male_shuanglangxiaoyang_cs_tob": "爽朗小阳(客服)",
        "ICL_zh_male_qingxinbobo_cs_tob": "清新波波(客服)",
        "ICL_zh_female_wenwanshanshan_cs_tob": "温婉珊珊(客服)",
        "ICL_zh_female_tianmeixiaoyu_cs_tob": "甜美小雨(客服)",
        "ICL_zh_female_reqingaina_cs_tob": "热情艾娜(客服)",
        "ICL_zh_female_tianmeixiaoju_cs_tob": "甜美小橘(客服)",
        "ICL_zh_male_chenwenmingzai_cs_tob": "沉稳明仔(客服)",
        "ICL_zh_male_qinqiexiaozhuo_cs_tob": "亲切小卓(客服)",
        "ICL_zh_female_lingdongxinxin_cs_tob": "灵动欣欣(客服)",
        "ICL_zh_female_guaiqiaokeer_cs_tob": "乖巧可儿(客服)",
        "ICL_zh_female_nuanxinqianqian_cs_tob": "暖心茜茜(客服)",
        "ICL_zh_female_ruanmengtuanzi_cs_tob": "软萌团子(客服)",
        "ICL_zh_male_yangguangyangyang_cs_tob": "阳光洋洋(客服)",
        "ICL_zh_female_ruanmengtangtang_cs_tob": "软萌糖糖(客服)",
        "ICL_zh_female_xiuliqianqian_cs_tob": "秀丽倩倩(客服)",
        "ICL_zh_female_kaixinxiaohong_cs_tob": "开心小鸿(客服)",
        "ICL_zh_female_qingyingduoduo_cs_tob": "轻盈朵朵(客服)",
        "zh_female_kefunvsheng_mars_bigtts": "暖阳女声(客服)",

        # ===== 视频配音 =====
        "zh_male_M100_conversation_wvae_bigtts": "悠悠君子",
        "zh_female_maomao_conversation_wvae_bigtts": "文静毛毛",
        "ICL_zh_female_qiuling_v1_tob": "倾心少女",
        "ICL_zh_male_buyan_v1_tob": "醇厚低音",
        "ICL_zh_male_BV144_paoxiaoge_v1_tob": "咆哮小哥",
        "ICL_zh_female_heainainai_tob": "和蔼奶奶",
        "ICL_zh_female_linjuayi_tob": "邻居阿姨",
        "zh_female_wenrouxiaoya_moon_bigtts": "温柔小雅",
        "zh_male_tiancaitongsheng_mars_bigtts": "天才童声",
        "zh_male_sunwukong_mars_bigtts": "猴哥",
        "zh_male_xionger_mars_bigtts": "熊二",
        "zh_female_peiqi_mars_bigtts": "佩奇猪",
        "zh_female_wuzetian_mars_bigtts": "武则天",
        "zh_female_gujie_mars_bigtts": "顾姐",
        "zh_female_yingtaowanzi_mars_bigtts": "樱桃丸子",
        "zh_male_chunhui_mars_bigtts": "广告解说",
        "zh_female_shaoergushi_mars_bigtts": "少儿故事",
        "zh_male_silang_mars_bigtts": "四郎",
        "zh_female_qiaopinvsheng_mars_bigtts": "俏皮女声",
        "zh_male_lanxiaoyang_mars_bigtts": "懒音绵宝",
        "zh_male_dongmanhaimian_mars_bigtts": "亮嗓萌仔",
        "zh_male_jieshuonansheng_mars_bigtts": "磁性解说男声/Morgan",
        "zh_female_jitangmeimei_mars_bigtts": "鸡汤妹妹/Hope",
        "zh_female_tiexinnvsheng_mars_bigtts": "贴心女声/Candy",
        "zh_female_mengyatou_mars_bigtts": "萌丫头/Cutey",

        # ===== 有声阅读 =====
        "ICL_zh_male_neiliancaijun_e991be511569_tob": "内敛才俊",
        "ICL_zh_male_yangyang_v1_tob": "温暖少年",
        "ICL_zh_male_flc_v1_tob": "儒雅公子",
        "zh_male_changtianyi_mars_bigtts": "悬疑解说",
        "zh_male_ruyaqingnian_mars_bigtts": "儒雅青年",
        "zh_male_baqiqingshu_mars_bigtts": "霸气青叔",
        "zh_male_qingcang_mars_bigtts": "擎苍",
        "zh_male_yangguangqingnian_mars_bigtts": "活力小哥",
        "zh_female_gufengshaoyu_mars_bigtts": "古风少御",
        "zh_female_wenroushunv_mars_bigtts": "温柔淑女",
        "zh_male_fanjuanqingnian_mars_bigtts": "反卷青年",

        # ===== 多语种 - 美式英语 =====
        "en_female_lauren_moon_bigtts": "Lauren",
        "en_male_campaign_jamal_moon_bigtts": "Energetic Male II",
        "en_male_chris_moon_bigtts": "Gotham Hero",
        "en_female_product_darcie_moon_bigtts": "Flirty Female",
        "en_female_emotional_moon_bigtts": "Peaceful Female",
        "en_female_nara_moon_bigtts": "Nara",
        "en_male_bruce_moon_bigtts": "Bruce",
        "en_male_michael_moon_bigtts": "Michael",
        "ICL_en_male_cc_sha_v1_tob": "Cartoon Chef",
        "zh_male_M100_conversation_wvae_bigtts": "Lucas",
        "en_female_sophie_conversation_wvae_bigtts": "Sophie",
        "en_female_dacey_conversation_wvae_bigtts": "Daisy",
        "en_male_charlie_conversation_wvae_bigtts": "Owen",
        "en_female_sarah_new_conversation_wvae_bigtts": "Luna",
        "ICL_en_male_michael_tob": "Michael",
        "ICL_en_female_cc_cm_v1_tob": "Charlie",
        "ICL_en_male_oogie2_tob": "Big Boogie",
        "ICL_en_male_frosty1_tob": "Frosty Man",
        "ICL_en_male_grinch2_tob": "The Grinch",
        "ICL_en_male_zayne_tob": "Zayne",
        "ICL_en_male_cc_jigsaw_tob": "Jigsaw",
        "ICL_en_male_cc_chucky_tob": "Chucky",
        "ICL_en_male_cc_penny_v1_tob": "Clown Man",
        "ICL_en_male_kevin2_tob": "Kevin McCallister",
        "ICL_en_male_xavier1_v1_tob": "Xavier",
        "ICL_en_male_cc_dracula_v1_tob": "Noah",
        "en_male_adam_mars_bigtts": "Adam",
        "en_female_amanda_mars_bigtts": "Amanda",
        "en_male_jackson_mars_bigtts": "Jackson",

        # ===== 多语种 - 英式英语 =====
        "en_female_daisy_moon_bigtts": "Delicate Girl",
        "en_male_dave_moon_bigtts": "Dave",
        "en_male_hades_moon_bigtts": "Hades",
        "en_female_onez_moon_bigtts": "Onez",
        "en_female_emily_mars_bigtts": "Emily",
        "zh_male_xudong_conversation_wvae_bigtts": "Daniel",
        "ICL_en_male_cc_alastor_tob": "Alastor",
        "en_male_smith_mars_bigtts": "Smith",
        "en_female_anna_mars_bigtts": "Anna",

        # ===== 多语种 - 澳洲英语 =====
        "ICL_en_male_aussie_v1_tob": "Ethan",
        "en_female_sarah_mars_bigtts": "Sarah",
        "en_male_dryw_mars_bigtts": "Dryw",

        # ===== 多语种 - 西语 =====
        "multi_female_maomao_conversation_wvae_bigtts": "Diana",
        "multi_male_M100_conversation_wvae_bigtts": "Lucía",
        "multi_female_sophie_conversation_wvae_bigtts": "Sofía",
        "multi_male_xudong_conversation_wvae_bigtts": "Daníel",

        # ===== 多语种 - 日语 =====
        "multi_zh_male_youyoujunzi_moon_bigtts": "ひかる（光）",
        "multi_female_sophie_conversation_wvae_bigtts": "さとみ（智美）",
        "multi_male_xudong_conversation_wvae_bigtts": "まさお（正男）",
        "multi_female_maomao_conversation_wvae_bigtts": "つき（月）",
        "multi_female_gaolengyujie_moon_bigtts": "あけみ（朱美）",
        "multi_male_jingqiangkanye_moon_bigtts": "かずね（和音）",
        "multi_female_shuangkuaisisi_moon_bigtts": "はるこ（晴子）",
        "multi_male_wanqudashu_moon_bigtts": "ひろし（広志）",
    }

    # 合并所有音色
    VOICES = {**VOICES_1_0, **VOICES_2_0}

    def __init__(
        self,
        app_id: str,
        access_key: str,
        resource_id: str = None,
        speaker: str = "zh_female_cancan_mars_bigtts",
        api_url: str = DEFAULT_URL,
    ):
        self.app_id = app_id
        self.access_key = access_key
        self.speaker = speaker
        self.api_url = api_url
        # Auto-detect resource_id based on speaker if not provided
        self.resource_id = resource_id or self._detect_resource_id(speaker)

    @classmethod
    def _detect_resource_id(cls, speaker: str) -> str:
        """根据音色自动检测 resource_id"""
        if speaker in cls.VOICES_2_0:
            return "seed-tts-2.0"
        return "seed-tts-1.0"

    def _get_headers(self) -> dict:
        """生成请求头"""
        return {
            "X-Api-App-Id": self.app_id,
            "X-Api-Access-Key": self.access_key,
            "X-Api-Resource-Id": self.resource_id,
            "Content-Type": "application/json",
            "Connection": "keep-alive",
        }

    def _build_payload(
        self,
        text: str,
        format: str = "mp3",
        sample_rate: int = 24000,
        speed: float = 1.0,
        enable_timestamp: bool = False,
        context_texts: list = None,
        emotion: str = None,
    ) -> dict:
        """构建请求参数"""
        additions = {
            "explicit_language": "zh",
            "disable_markdown_filter": True,
        }
        if enable_timestamp:
            additions["enable_timestamp"] = True

        req_params = {
            "text": text,
            "speaker": self.speaker,
            "audio_params": {
                "format": format,
                "sample_rate": sample_rate,
                "enable_timestamp": enable_timestamp,
                "speed": speed,
            },
            "additions": json.dumps(additions),
        }

        # Only add context_texts for 2.0 speakers (only the first value is effective)
        if context_texts and self.resource_id == "seed-tts-2.0":
            # context_texts should be a list of strings, e.g., ["你可以说慢一点吗？"]
            if isinstance(context_texts, list) and len(context_texts) > 0:
                req_params["context_texts"] = context_texts

        # Add emotion parameter if provided (only supported by certain speakers)
        if emotion:
            req_params["audio_params"]["emotion"] = emotion

        return {
            "user": {"uid": "xiaozhi_user"},
            "req_params": req_params,
        }

    def synthesize(
        self,
        text: str,
        format: str = "mp3",
        sample_rate: int = 24000,
        speed: float = 1.0,
        context_texts: list = None,
        emotion: str = None,
    ) -> Optional[bytes]:
        """
        合成语音，返回音频数据

        Args:
            text: 要合成的文本
            format: 音频格式 (mp3, wav, opus)
            sample_rate: 采样率
            speed: 语速 (0.8-2.0)
            emotion: 情感参数 (如: happy, sad, angry, excited 等)

        Returns:
            PCM 音频数据 (bytes) 或 None (失败)
        """
        headers = self._get_headers()
        payload = self._build_payload(text, format, sample_rate, speed, context_texts=context_texts, emotion=emotion)

        print(f"[DoubaoTTS] Request: speaker={self.speaker}, resource_id={self.resource_id}")

        session = requests.Session()
        response = None
        try:
            response = session.post(
                self.api_url, headers=headers, json=payload, stream=True, timeout=60
            )
            response.raise_for_status()

            audio_data = bytearray()

            for chunk in response.iter_lines(decode_unicode=True):
                if not chunk:
                    continue

                data = json.loads(chunk)

                # 音频数据
                if data.get("code", 0) == 0 and "data" in data and data["data"]:
                    chunk_audio = base64.b64decode(data["data"])
                    audio_data.extend(chunk_audio)
                    continue

                # 句子信息
                if data.get("code", 0) == 0 and "sentence" in data and data["sentence"]:
                    continue

                # 完成
                if data.get("code", 0) == 20000000:
                    break

                # 错误
                if data.get("code", 0) > 0:
                    print(f"TTS Error: {data}")
                    print(f"TTS Request Info: resource_id={self.resource_id}, speaker={self.speaker}")
                    raise Exception(f"TTS API Error {data.get('code')}: {data.get('message')}")

            if not audio_data:
                return None

            # 解码为 PCM
            return self._decode_to_pcm(bytes(audio_data), format, target_sample_rate=24000)

        except Exception as e:
            print(f"TTS request failed: {e}")
            return None
        finally:
            if response:
                response.close()
            session.close()

    def synthesize_stream(
        self,
        text: str,
        format: str = "mp3",
        sample_rate: int = 24000,
        speed: float = 1.0,
        context_texts: list = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        流式合成语音

        Yields:
            PCM 音频数据块
        """
        headers = self._get_headers()
        payload = self._build_payload(text, format, sample_rate, speed, context_texts=context_texts)

        session = requests.Session()
        response = None
        buffer = bytearray()

        try:
            response = session.post(
                self.api_url, headers=headers, json=payload, stream=True, timeout=60
            )
            response.raise_for_status()

            for chunk in response.iter_lines(decode_unicode=True):
                if not chunk:
                    continue

                data = json.loads(chunk)

                if data.get("code", 0) == 0 and "data" in data and data["data"]:
                    chunk_audio = base64.b64decode(data["data"])
                    buffer.extend(chunk_audio)

                    # 积累一定数据后解码并yield
                    if len(buffer) > 8192:
                        pcm = self._decode_to_pcm(bytes(buffer), format, target_sample_rate=24000)
                        if pcm:
                            yield pcm
                        buffer.clear()

                elif data.get("code", 0) == 20000000:
                    # 最后一块
                    if buffer:
                        pcm = self._decode_to_pcm(bytes(buffer), format, target_sample_rate=24000)
                        if pcm:
                            yield pcm
                    break

        except Exception as e:
            print(f"TTS stream error: {e}")
        finally:
            if response:
                response.close()
            session.close()

    def _decode_to_pcm(
        self, audio_data: bytes, format: str, target_sample_rate: int = 24000
    ) -> Optional[bytes]:
        """将音频数据解码为 PCM 格式"""
        try:
            # 使用 soundfile 读取音频
            with io.BytesIO(audio_data) as bio:
                data, sr = sf.read(bio, dtype="float32")

            # 转为单声道
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            # 重采样到目标采样率
            if sr != target_sample_rate:
                # 计算重采样比例
                gcd = np.gcd(sr, target_sample_rate)
                up = target_sample_rate // gcd
                down = sr // gcd
                data = signal.resample_poly(data, up, down)

            # 转为 16-bit int
            data = np.clip(data, -1.0, 1.0)
            data = (data * 32767).astype(np.int16)

            return data.tobytes()

        except Exception as e:
            print(f"Decode audio error: {e}")
            return None

    @classmethod
    def list_voices(cls) -> dict:
        """返回可用音色列表"""
        return cls.VOICES.copy()

    @classmethod
    def list_voices_by_version(cls, version: str = "all") -> dict:
        """按版本返回可用音色列表

        Args:
            version: "1.0", "2.0", or "all"
        """
        if version == "1.0":
            return cls.VOICES_1_0.copy()
        elif version == "2.0":
            return cls.VOICES_2_0.copy()
        return cls.VOICES.copy()
