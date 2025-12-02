"""
AI doctor-patient diagnostic system - with complete records and long-term learning mechanism
"""

import random
import time
import json
import os
import re
import sys
from datetime import datetime
from typing import List, Dict, Optional
from colorama import Fore, Style, init
from openai import OpenAI
from dotenv import load_dotenv

# 初始化colorama
init(autoreset=True)

# 加载环境变量
load_dotenv()


# ==================== 配置类 ====================

class MedicalConfig:
    """Medical configuration class"""
    
    # ==================== API配置 ====================
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    MODEL_NAME = "deepseek-chat"

    # ====================基础配置 ====================
    MAX_QUESTIONS_PER_ROUND = 12  # 每轮最多问题数
    INITIAL_BUDGET = 500  # 初始预算
    SUSPICION_THRESHOLD = 0.8  # 怀疑阈值
    
    # ==================== 显示配置 ====================
    SHOW_AI_THINKING = True  # 显示AI思考过程
    SHOW_DETAILED_LOGS = True  # 显示详细日志
    
    # ==================== 记录配置 ====================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_RECORDS = True  # 保存记录
    RECORDS_DIRC = os.path.join(BASE_DIR, "medical_records")
    DOCTOR_MEMORY_DIR = os.path.join(BASE_DIR, "doctor_memory") 
    ROUND_LOGS_DIR = os.path.join(BASE_DIR, "round_logs")
    ENABLE_LONG_TERM_MEMORY = True  # 启用长期记忆
    MAX_HISTORY = 10  # 保存最近10场记录
    
    # ==================== 费用配置 ====================
    QUESTION_COST = 0  # 询问不收费
    TEST_COSTS = {
    "血常规": 80,
    "尿常规": 60, 
    "心电图": 120,
    "X光胸片": 150,
    "CT扫描": 300,
    "MRI": 500,
    "超声检查": 200,
    "胃镜检查": 400,
    # 新增项目
    "肝功能检查": 90,
    "肾功能检查": 85,
    "血糖检测": 50,
    "血脂分析": 110,
    "骨密度检查": 180,
    "内窥镜检查": 350,
    "病理活检": 250,
    "脑电图": 160,
    "肺功能检查": 130,
    "皮肤过敏测试": 95
    }
    
    TEST_ACCURACY = {
    "血常规": 0.7,
    "尿常规": 0.65,
    "心电图": 0.8,
    "X光胸片": 0.75,
    "CT扫描": 0.9,
    "MRI": 0.95,
    "超声检查": 0.85,
    "胃镜检查": 0.88,
    # 新增项目
    "肝功能检查": 0.72,
    "肾功能检查": 0.68,
    "血糖检测": 0.95,
    "血脂分析": 0.82,
    "骨密度检查": 0.88,
    "内窥镜检查": 0.92,
    "病理活检": 0.96,
    "脑电图": 0.78,
    "肺功能检查": 0.85,
    "皮肤过敏测试": 0.9
    }


    # ==================== AI参数配置 ====================
    # 温度参数 - 不同场景使用不同温度
    TEMPERATURE_PATIENT_RESPONSE = 0.9    # 患者回答 - 高温度增加多样性
    TEMPERATURE_DOCTOR_QUESTION = 0.7     # 医生提问 - 中等温度平衡专业和灵活
    TEMPERATURE_DOCTOR_DIAGNOSIS = 0.3    # 医生诊断 - 低温度确保准确性
    TEMPERATURE_CASE_GENERATION = 0.6     # 病例生成 - 中等温度保证真实性
    
    MAX_TOKENS = 800

    # ==================== 疾病库 ====================
    DISEASE_LIBRARY = [
    "偏头痛", "胃炎", "过敏性鼻炎", "普通感冒", "高血压", 
    "糖尿病", "哮喘", "关节炎", "皮肤病", "失眠症",
    # 新增疾病
    "肺炎", "支气管炎", "胃溃疡", "肾结石", "胆囊炎",
    "心肌炎", "脑震荡", "腰椎间盘突出", "骨质疏松", "贫血",
    "甲状腺功能亢进", "痛风", "肝炎", "肠易激综合征", "抑郁症",
    "焦虑症", "白内障", "青光眼", "中耳炎", "鼻窦炎"
    ]

    # ==================== 患者个性类型 ====================
    PERSONALITY_TYPES = {
    "谨慎型": {"suspicion_gain": 0.15, "cost_sensitivity": 0.8, "ideal_cost_range": (160, 300)},
    "随意型": {"suspicion_gain": 0.08, "cost_sensitivity": 0.4, "ideal_cost_range": (240, 400)},
    "疑病症": {"suspicion_gain": 0.25, "cost_sensitivity": 0.3, "ideal_cost_range": (300, 500)},
    "节俭型": {"suspicion_gain": 0.12, "cost_sensitivity": 0.9, "ideal_cost_range": (100, 200)},
    # 新增个性类型
    "急躁型": {"suspicion_gain": 0.20, "cost_sensitivity": 0.5, "ideal_cost_range": (200, 350)},
    "依赖型": {"suspicion_gain": 0.05, "cost_sensitivity": 0.6, "ideal_cost_range": (400, 600)},
    "理性型": {"suspicion_gain": 0.10, "cost_sensitivity": 0.7, "ideal_cost_range": (300, 440)},
    "多疑型": {"suspicion_gain": 0.30, "cost_sensitivity": 0.4, "ideal_cost_range": (160, 240)}
    }

    # ==================== 误解触发器 ====================
    MISUNDERSTANDING_TRIGGERS = {
    "吃饭": {"threshold": 0.4, "misunderstanding": "认为几小时前吃饭的算'空腹'"},
    "喝酒": {"threshold": 0.3, "misunderstanding": "不认为啤酒算'喝酒'"},
    "运动": {"threshold": 0.5, "misunderstanding": "认为散步不算'运动'"},
    "睡眠": {"threshold": 0.4, "misunderstanding": "把打盹也算作'睡觉'"},
    "疼痛": {"threshold": 0.6, "misunderstanding": "分不清酸痛和刺痛"},
    # 新增触发器
    "恶心": {"threshold": 0.35, "misunderstanding": "把胃部不适说成恶心"},
    "头晕": {"threshold": 0.45, "misunderstanding": "分不清头晕和眩晕"},
    "发热": {"threshold": 0.3, "misunderstanding": "把正常体温波动当发烧"},
    "咳嗽": {"threshold": 0.4, "misunderstanding": "把清嗓子也算作咳嗽"},
    "乏力": {"threshold": 0.5, "misunderstanding": "把正常疲劳说成病态乏力"},
    "食欲": {"threshold": 0.35, "misunderstanding": "把心情不好说成没食欲"},
    "药物": {"threshold": 0.4, "misunderstanding": "忘记用药或记错剂量"},
    "时间": {"threshold": 0.6, "misunderstanding": "记错症状开始时间"},
    "频率": {"threshold": 0.55, "misunderstanding": "夸大或缩小症状频率"},
    "位置": {"threshold": 0.5, "misunderstanding": "描述不准疼痛位置"}
    }

    @classmethod
    def validate(cls):
        """验证配置有效性"""
        if not cls.DEEPSEEK_API_KEY:
            raise ValueError(
                "❌ 错误: 未找到DEEPSEEK_API_KEY!\n"
                "请在.env文件中设置DEEPSEEK_API_KEY=your_api_key\n"
                "或设置环境变量: export DEEPSEEK_API_KEY=your_api_key"
            )
        
        # 创建记录目录
        if cls.SAVE_RECORDS:
            os.makedirs(cls.RECORDS_DIRC, exist_ok=True)
            os.makedirs(cls.DOCTOR_MEMORY_DIR, exist_ok=True)
            os.makedirs(cls.ROUND_LOGS_DIR, exist_ok=True)
            
        print("✅ 医疗配置验证成功")
        return True


# ==================== 记忆管理系统 ====================

class MemoryManager:
    """记忆管理器 - 处理医生的长期学习记忆"""
    
    def __init__(self):
        self.memory_dir = MedicalConfig.DOCTOR_MEMORY_DIR
        self.memory_file = os.path.join(self.memory_dir, "doctor_memory.json")
        os.makedirs(self.memory_dir, exist_ok=True)
    
    def save_learning_experience(self, experience: Dict, run_id: str):
        """保存学习经验到长期记忆"""
        memories = self._load_memory()
        
        memories.append({
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "experience": experience
        })
        
        # 限制记忆数量
        if len(memories) > MedicalConfig.MAX_HISTORY:
            memories = memories[-MedicalConfig.MAX_HISTORY:]
            
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(memories, f, ensure_ascii=False, indent=2)
    
    def load_learning_experience(self) -> str:
        """加载长期学习经验"""
        memories = self._load_memory()
        
        if not memories:
            return "暂无历史学习经验"
        
        experience_parts = []
        experience_parts.append("【医生历史学习经验】")
        experience_parts.append(f"(基于最近{len(memories)}场的经验总结)")
        
        for i, memory in enumerate(memories[-5:], 1):  # 显示最近5场
            exp = memory['experience']
            exp_summary = f"诊断{i}: 成功率{exp.get('success_rate', 0):.1%}, 平均问题{exp.get('avg_questions', 0):.1f}, 关键学习: {exp.get('key_learning', '')}"
            experience_parts.append(exp_summary)
        
        return "\n".join(experience_parts)
    
    def _load_memory(self) -> list:
        """加载记忆文件"""
        if not os.path.exists(self.memory_file):
            return []
        
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []


# ==================== 记录系统 ====================

class RecordManager:
    """记录管理器 - 处理记录和回合日志"""
    
    def __init__(self):
        self.RECORDS_DIRC = MedicalConfig.RECORDS_DIRC
        self.round_logs_dir = MedicalConfig.ROUND_LOGS_DIR
    
    def save_program_record(self, program_data: Dict) -> str:
        """保存完整记录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"program_{timestamp}.json"
        filepath = os.path.join(self.RECORDS_DIRC, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(program_data, f, ensure_ascii=False, indent=2)
        
        return timestamp
    
    def save_round_log(self, round_data: Dict, round_number: int) -> str:
        """保存单轮详细日志"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"round_{round_number}_{timestamp}.json"
        filepath = os.path.join(self.round_logs_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(round_data, f, ensure_ascii=False, indent=2)
        
        return filepath


# ==================== API客户端 ====================

class DeepSeekClient:
    """DeepSeek API客户端类"""

    def __init__(self):
        """初始化DeepSeek客户端"""
        self.client = OpenAI(
            api_key=MedicalConfig.DEEPSEEK_API_KEY,
            base_url=MedicalConfig.DEEPSEEK_BASE_URL
        )
        self.model = MedicalConfig.MODEL_NAME
        self.max_tokens = MedicalConfig.MAX_TOKENS

    def chat(self, system_prompt: str, user_message: str, temperature: float = 0.7) -> str:
        """发送聊天请求到DeepSeek API"""
        try:
            start_time = time.time()

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=self.max_tokens
            )

            elapsed_time = time.time() - start_time
            reply = response.choices[0].message.content

            if MedicalConfig.SHOW_AI_THINKING:
                print(f"⏱️  API响应时间: {elapsed_time:.2f}s")
            return reply

        except Exception as e:
            error_msg = f"❌ DeepSeek API调用失败: {str(e)}"
            print(error_msg)
            # 返回降级响应
            return "我需要更多信息来判断您的情况。"


# ==================== 医疗系统 ====================

class MedicalSystem:
    """医疗系统 - 处理检查执行和费用计算"""
    TEST_DISEASE_RELEVANCE = {
    # ==================== 血液/生化检查 ====================
    "血糖检测": {
        "糖尿病": 0.95,        # 直接诊断依据
        "高血压": 0.25,        # 可能伴随糖代谢异常
        "甲状腺功能亢进": 0.20,  # 可能影响血糖
        "普通感冒": 0.05,      # 基本无关
        "胃炎": 0.10,          # 胃病可能影响进食，间接相关
        "肺炎": 0.10,          # 感染可能引起应激性高血糖
    },
    
    "糖化血红蛋白": {
        "糖尿病": 0.90,        # 反映长期血糖控制
        "贫血": 0.40,          # 影响HbA1c测量
        "肾病": 0.35,          # 肾功能影响HbA1c
    },
    
    "血常规": {
        "感染性疾病": 0.80,    # 白细胞计数是感染标志
        "肺炎": 0.75,
        "普通感冒": 0.65,
        "支气管炎": 0.70,
        "贫血": 0.85,          # 血红蛋白是直接指标
        "白血病": 0.90,
        "胃炎": 0.40,          # 可能伴随慢性失血
        "糖尿病": 0.30,        # 可能并发感染
        "偏头痛": 0.10,        # 基本无关
    },
    
    "肝功能检查": {
        "肝炎": 0.90,
        "肝硬化": 0.85,
        "胆囊炎": 0.60,
        "药物性肝损伤": 0.80,
        "糖尿病": 0.25,        # 可能并发脂肪肝
        "高血压": 0.15,
    },
    
    "肾功能检查": {
        "肾病": 0.90,
        "肾结石": 0.70,
        "高血压": 0.60,        # 高血压肾病
        "糖尿病": 0.65,        # 糖尿病肾病
        "痛风": 0.50,          # 可能影响肾功能
    },
    
    "血脂分析": {
        "高血压": 0.60,        # 常合并血脂异常
        "糖尿病": 0.65,        # 常合并血脂异常
        "心脏病": 0.70,        # 冠心病风险因素
        "动脉硬化": 0.75,
    },
    
    # ==================== 影像学检查 ====================
    "X光胸片": {
        "肺炎": 0.85,          # 可见肺部浸润影
        "肺结核": 0.80,        # 可见结核病灶
        "支气管炎": 0.50,      # 可能仅纹理增粗
        "心脏病": 0.65,        # 可见心影增大
        "肺癌": 0.70,
        "骨折": 0.95,          # 骨折直接可见
        "胃炎": 0.05,          # 基本看不见胃
        "糖尿病": 0.01,        # 完全无关
    },
    
    "CT扫描": {
        "肺炎": 0.90,          # 比X光更敏感
        "脑震荡": 0.70,        # 排除颅内出血
        "骨折": 0.95,
        "脑肿瘤": 0.85,
        "腰椎间盘突出": 0.90,
        "肾结石": 0.95,        # 尿路结石
        "胃炎": 0.30,          # 可显示胃壁增厚
        "心脏病": 0.60,        # 冠脉CT
    },
    
    "MRI": {
        "脑震荡": 0.75,        # 比CT对脑组织更敏感
        "脑肿瘤": 0.95,
        "腰椎间盘突出": 0.95,
        "关节炎": 0.85,        # 关节软组织
        "心肌炎": 0.80,        # 心脏MRI
        "肺炎": 0.60,          # 可用但非首选
    },
    
    "超声检查": {
        "胆囊炎": 0.90,        # 胆囊壁增厚、结石
        "肾结石": 0.85,
        "肝硬化": 0.80,        # 肝脏形态
        "甲状腺功能亢进": 0.75,  # 甲状腺大小、血流
        "心脏病": 0.70,        # 心脏超声
        "肺炎": 0.40,          # 胸腔积液可见
        "胃炎": 0.30,          # 可排除其他腹部疾病
    },
    
    # ==================== 心电检查 ====================
    "心电图": {
        "心脏病": 0.90,        # 心律失常、心肌缺血
        "心肌炎": 0.85,
        "高血压": 0.60,        # 左室肥厚表现
        "甲状腺功能亢进": 0.50,  # 可能心动过速
        "糖尿病": 0.20,        # 可能并发冠心病
        "肺炎": 0.25,          # 可能继发心脏影响
        "胃炎": 0.05,          # 基本无关
        "偏头痛": 0.05,
    },
    
    "动态心电图": {
        "心脏病": 0.95,        # 捕捉阵发性心律失常
        "晕厥": 0.85,          # 心源性晕厥
        "心悸": 0.90,
        "心肌炎": 0.80,
    },
    
    # ==================== 内窥镜检查 ====================
    "胃镜检查": {
        "胃炎": 0.95,          # 直接观察胃黏膜
        "胃溃疡": 0.90,
        "胃癌": 0.85,          # 可活检
        "食管炎": 0.80,
        "糖尿病": 0.15,        # 可能胃轻瘫，但非首选
        "肝炎": 0.05,          # 基本无关
    },
    
    "肠镜检查": {
        "肠炎": 0.90,
        "结肠癌": 0.95,
        "肠易激综合征": 0.30,  # 排除性诊断
        "胃炎": 0.10,          # 不同部位
    },
    
    # ==================== 特殊检查 ====================
    "肺功能检查": {
        "哮喘": 0.95,          # 阻塞性通气功能障碍
        "支气管炎": 0.85,
        "肺炎": 0.50,          # 限制性可能
        "心脏病": 0.30,        # 心功能不全影响
        "糖尿病": 0.10,
    },
    
    "骨密度检查": {
        "骨质疏松": 0.95,      # 直接测量骨密度
        "骨折": 0.60,          # 评估骨折风险
        "关节炎": 0.40,
        "甲状腺功能亢进": 0.50,  # 可能骨代谢异常
    },
    
    "脑电图": {
        "癫痫": 0.90,
        "脑炎": 0.75,
        "偏头痛": 0.40,        # 有时做排除诊断
        "脑震荡": 0.30,
        "失眠症": 0.50,        # 睡眠脑电
    },
    
    "过敏测试": {
        "过敏性鼻炎": 0.95,
        "哮喘": 0.85,          # 过敏性哮喘
        "皮肤病": 0.80,        # 过敏性皮炎
        "食物过敏": 0.90,
    },
}

    def __init__(self):
        self.test_costs = MedicalConfig.TEST_COSTS
        self.test_accuracy = MedicalConfig.TEST_ACCURACY

    def perform_test(self, test_name: str, true_condition: str) -> Dict:
        """执行检查并返回结果"""
        cost = self.test_costs[test_name]
        base_accuracy = self.test_accuracy[test_name]
        
        # 获取检查对该疾病的相关性
        relevance = self.TEST_DISEASE_RELEVANCE.get(test_name, {}).get(true_condition, 0.1)
        
        # 最终准确率 = 基础准确率 × 相关性
        final_accuracy = base_accuracy * relevance
        
        # 决定检查结果
        if random.random() < final_accuracy:
            # ✅ 真阳性：检查正确发现了疾病
            return {
                "result": self._get_positive_result(test_name, true_condition),
                "cost": cost,
                "accurate": True,
                "relevance": relevance,  # 新增：记录相关性
                "result_type": "true_positive"
            }
        else:
            # 假阴性或正常结果
            if relevance < 0.3:
                # 🔍 低相关性检查：返回正常结果（本来就不太可能阳性）
                return {
                    "result": self._get_normal_result(test_name),
                    "cost": cost,
                    "accurate": True,  # 这实际上是"正确的阴性"
                    "relevance": relevance,
                    "result_type": "true_negative"  # 真阴性
                }
            else:
                # ❌ 假阴性：相关检查但漏诊了
                return {
                    "result": self._get_false_negative_result(test_name, true_condition),
                    "cost": cost,
                    "accurate": False,
                    "relevance": relevance,
                    "result_type": "false_negative"  # 假阴性
                }
    
    def _get_positive_result(self, test_name: str, disease: str) -> str:
        """生成阳性结果描述"""
        templates = {
            "血糖检测": f"血糖检测显示血糖明显升高，符合{disease}诊断标准",
            "心电图": f"心电图显示异常波形，提示{disease}可能",
            "X光胸片": f"X光胸片显示肺部阴影，符合{disease}表现",
            "血常规": f"血常规检查多项指标异常，支持{disease}诊断"
        }
        return templates.get(test_name, f"{test_name}检查显示异常，与{disease}相关")
    
    def _get_false_negative_result(self, test_name: str, disease: str) -> str:
        """生成假阴性结果描述"""
        false_negatives = {
            "糖尿病": {
                "血糖检测": "血糖值在正常范围上限，建议复查",
                "血常规": "血常规检查无明显异常"
            },
            "肺炎": {
                "X光胸片": "X光胸片未见明显肺部阴影",
                "血常规": "白细胞计数轻度升高，无特异性"
            },
            # ... 其他疾病的假阴性描述
        }
        
        return false_negatives.get(disease, {}).get(
            test_name, 
            f"{test_name}检查结果在正常范围内"
        )
    
    def _get_normal_result(self, test_name: str) -> str:
        """生成正常结果描述（用于低相关性检查）"""
        normal_results = {
            "心电图": "心电图显示正常窦性心律",
            "血糖检测": "血糖值在正常范围内",
            "X光胸片": "胸部X光片未见明显异常",
            "血常规": "血常规各项指标均在正常范围"
        }
        return normal_results.get(test_name, f"{test_name}检查未见异常")

    def get_available_tests(self) -> List[str]:
        """获取可用检查项目"""
        return list(self.test_costs.keys())


# ==================== 状态管理 ====================

class programState:
    """状态管理类"""

    def __init__(self):
        self.current_round = 0
        self.total_cost = 0
        self.remaining_budget = MedicalConfig.INITIAL_BUDGET
        self.questions_asked = 0
        self.tests_ordered = 0
        self.patient_suspicion = 0.0
        self.actions_history = []
        self.dialogue_history = []
        self.test_results = []
        self.start_time = datetime.now()
        self.patient_symptoms = []
        self.evidence_sufficient = False

    def record_action(self, action_type: str, details: Dict):
        """记录行动历史"""
        action = {
            "round": self.current_round,
            "type": action_type,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.actions_history.append(action)

    def add_question(self):
        """增加问题计数"""
        self.questions_asked += 1
        self.patient_suspicion += 0.1  # 每个问题增加怀疑值

    def add_test(self, cost: int):
        """增加检查计数和费用"""
        self.tests_ordered += 1
        self.total_cost += cost
        self.remaining_budget -= cost
        self.patient_suspicion += 0.15 

    def is_round_over(self, doctor_agent=None) -> bool:
        """检查回合是否结束"""
        # 基本结束条件
        basic_over = (self.patient_suspicion >= MedicalConfig.SUSPICION_THRESHOLD or
                     self.remaining_budget <= 0 or
                     self.questions_asked >= MedicalConfig.MAX_QUESTIONS_PER_ROUND)
        
        # 如果基本条件已满足，直接返回
        if basic_over:
            return True
        
        # 如果有医生智能体，询问是否证据充分
        if doctor_agent and self.questions_asked >= 3:  # 至少问3个问题后才可能证据充分
            # 更新证据充分标志
            self.evidence_sufficient = doctor_agent.is_evidence_sufficient(
                self.dialogue_history, 
                self.test_results,
                self.current_round,
                self.patient_suspicion
            )
            
            # 如果医生认为证据充分，回合结束
            if self.evidence_sufficient:
                print(f"🧠 医生认为证据充分，准备进行诊断")
                return True
        
        return False

    def get_status_summary(self) -> str:
        """获取状态摘要"""
        evidence_status = "✅证据充分" if self.evidence_sufficient else "📝采集中"
        return (f"当前回合: {self.current_round} | "
                f"问题数: {self.questions_asked} | "
                f"检查数: {self.tests_ordered} | "
                f"总费用: {self.total_cost} | "
                f"剩余预算: {self.remaining_budget} | "
                f"患者怀疑: {self.patient_suspicion:.2f} | "
                f"{evidence_status}")
    
    def export_to_dict(self) -> Dict:
        """导出状态为字典"""
        return {
            "current_round": self.current_round,
            "total_cost": self.total_cost,
            "remaining_budget": self.remaining_budget,
            "questions_asked": self.questions_asked,
            "tests_ordered": self.tests_ordered,
            "patient_suspicion": self.patient_suspicion,
            "actions_history": self.actions_history,
            "dialogue_history": self.dialogue_history,
            "test_results": self.test_results,
            "patient_symptoms": self.patient_symptoms,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat()
        }


# ==================== 智能体类 ====================

class PatientAgent:
    """患者智能体"""

    def __init__(self, api_client: DeepSeekClient, case_info: Dict):
        self.api_client = api_client
        self.true_condition = case_info["true_disease"]
        self.symptoms_description = case_info["symptoms_description"]
        self.personality = case_info["personality"]
        self.ideal_cost = case_info["ideal_cost"]
        self.suspicion_level = 0.0
        self.dialogue_history = []

    def respond_to_question(self, question: str) -> str:
        """回答医生问题（可能不准确）"""
        # 增加怀疑值
        suspicion_gain = MedicalConfig.PERSONALITY_TYPES[self.personality]["suspicion_gain"]
        self.suspicion_level += suspicion_gain

        # 判断是否产生误解
        if self._should_misunderstand(question):
            return self._generate_misunderstanding_response(question)
        else:
            return self._generate_truthful_response(question)

    def _should_misunderstand(self, question: str) -> bool:
        """判断是否对问题产生误解"""
        for trigger, info in MedicalConfig.MISUNDERSTANDING_TRIGGERS.items():
            if trigger in question and random.random() < info["threshold"]:
                return True
        return False

    def _generate_misunderstanding_response(self, question: str) -> str:
        """生成误解回答"""
        prompt = f"""你是患者，现在医生问你: "{question}"

你的真实病情: {self.symptoms_description}

请基于你的真实情况，但产生一些误解来回答：
- 可以理解错误医生的意思
- 可以记错或混淆一些细节
- 保持自然、口语化
- 不超过50字"""

        response = self.api_client.chat(
            system_prompt="你是一个患者，有时会误解医生的问题",
            user_message=prompt,
            temperature=MedicalConfig.TEMPERATURE_PATIENT_RESPONSE
        )
        return response

    def _generate_truthful_response(self, question: str) -> str:
        """生成真实回答"""
        prompt = f"""你是患者，现在医生问你: "{question}"

你的真实病情: {self.symptoms_description}

请基于真实情况回答医生：
- 准确描述你的感受
- 可以有些不确定但不要故意误导
- 保持自然、口语化
- 不超过50字"""

        response = self.api_client.chat(
            system_prompt="你是一个诚实的患者，正在向医生描述病情",
            user_message=prompt,
            temperature=MedicalConfig.TEMPERATURE_PATIENT_RESPONSE
        )
        return response

    def get_initial_complaint(self) -> str:
        """获取初始主诉"""
        prompt = f"""你是患者，现在要向医生描述你的不适。

你的病情: {self.symptoms_description}

请用自然的口语描述你的症状：
- 开头像真实患者那样自然描述
- 可以有些模糊表达（如"好像"、"有点"、"说不清"）
- 不超过80字"""

        response = self.api_client.chat(
            system_prompt="你是一个身体不适的患者，正在向医生描述病情",
            user_message=prompt,
            temperature=MedicalConfig.TEMPERATURE_PATIENT_RESPONSE
        )

        self.dialogue_history.append({
            "role": "patient",
            "content": response
        })

        return response
    
    def export_to_dict(self) -> Dict:
        """导出患者信息为字典"""
        return {
            "true_condition": self.true_condition,
            "symptoms_description": self.symptoms_description,
            "personality": self.personality,
            "ideal_cost": self.ideal_cost,
            "final_suspicion": self.suspicion_level,
            "dialogue_history": self.dialogue_history
        }


class DoctorAgent:
    """医生智能体"""

    def __init__(self, api_client: DeepSeekClient):
        self.api_client = api_client
        self.learning_history = []
        self.consultation_log = []
        self.successful_strategies = {}
        self.memory_manager = MemoryManager()
        self.historical_experience = ""
        self.confidence_threshold = 0.8
        
        # 加载长期记忆
        if MedicalConfig.ENABLE_LONG_TERM_MEMORY:
            self.historical_experience = self.memory_manager.load_learning_experience()
            if self.historical_experience:
                print(f"✅ 医生加载了长期记忆经验")
    def is_evidence_sufficient(self, dialogue_history: List, test_results: List, 
                              current_round: int, current_suspicion: float) -> bool:
        """判断证据是否足够进行诊断"""
        
        # 如果有检查结果，构造检查结果摘要
        test_summary = ""
        if test_results:
            test_summary = f"【已做检查】{len(test_results)}项检查：{', '.join([r.split(':')[0] for r in test_results if ':' in r][:3])}"
        
        # 获取最近对话（最后4条）
        recent_dialogue = dialogue_history[-6:] if len(dialogue_history) >= 6 else dialogue_history
        dialogue_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_dialogue])
        
        prompt = f"""作为经验丰富的医生，你需要判断当前收集的证据是否足够做出诊断。

【当前问诊情况】
- 当前回合: {current_round}
- 患者怀疑度: {current_suspicion:.2f}
{test_summary}

【最近对话记录】
{dialogue_text}

请评估：
1. 关键症状是否已明确？
2. 关键鉴别检查是否已完成？
3. 是否有足够证据排除其他可能疾病？
4. 能否以较高置信度做出诊断？

如果证据足够，请回答"是的，证据足够诊断"。
如果还需要更多信息，请回答"不，需要更多信息"。

只回答上述两个选项之一："""
        
        try:
            response = self.api_client.chat(
                system_prompt="你是经验丰富的临床医生，善于判断何时可以做出诊断",
                user_message=prompt,
                temperature=0.3  # 低温度确保判断稳定
            ).strip()
            
            # 判断响应
            if "是的，证据足够诊断" in response or "证据足够" in response:
                return True
            elif "不，需要更多信息" in response or "需要更多信息" in response:
                return False
            else:
                # 如果响应不明确，根据对话长度和检查数量判断
                has_tests = len(test_results) > 0
                sufficient_dialogue = len(dialogue_history) >= 6
                return (has_tests and sufficient_dialogue) or len(dialogue_history) >= 10
                
        except Exception as e:
            print(f"⚠️ 证据评估API调用失败: {e}")
            # 降级策略：基于简单规则
            return len(dialogue_history) >= 8 or (len(test_results) >= 2 and len(dialogue_history) >= 4)
    def choose_action(self, program_state: programState, patient: PatientAgent) -> str:
        """选择行动：询问病情 或 要求检查"""
        # 基于学习历史的策略
        suspicion = patient.suspicion_level
        budget_ratio = program_state.remaining_budget / MedicalConfig.INITIAL_BUDGET
        
        # 简单策略：基于怀疑值和预算决定
        if (suspicion > 0.6 and budget_ratio > 0.3) or suspicion > 0.8:
            return "要求检查"
        else:
            return "询问病情"

    def generate_question(self, dialogue_history: List) -> str:
        """生成诊断问题"""
        history_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in dialogue_history[-4:]  # 最近2轮对话
        ]) if dialogue_history else "暂无对话历史"

        prompt = f"""你是一个经验丰富的医生，正在诊断患者。

【当前对话历史】
{history_text}

{self.historical_experience if self.historical_experience else ''}

请提出一个最有助于诊断的问题：
- 要基于已有信息推理
- 问题要精准、有针对性
- 单次只问一个问题

输出问题："""

        question = self.api_client.chat(
            system_prompt="你是一个专业的医生，善于通过问诊诊断疾病",
            user_message=prompt,
            temperature=MedicalConfig.TEMPERATURE_DOCTOR_QUESTION
        )
        return question.strip()

    def select_test_type(self, program_state: programState, symptoms: List[str], dialogue_history: List) -> str:
        """根据患者病情，从检查列表中选择最合适的检查"""
        
        # 获取所有检查项目
        available_tests = list(MedicalConfig.TEST_COSTS.keys())
        
        # 如果预算不足或没有症状，返回一个基础检查
        if program_state.remaining_budget < 50 or not symptoms:
            return self._select_basic_test(program_state.remaining_budget)
        
        # 构建症状描述
        symptoms_text = "、".join(symptoms) if symptoms else "全身不适"
        
        # 获取近期对话
        recent_dialogue = dialogue_history[-4:] if len(dialogue_history) >= 4 else dialogue_history
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_dialogue])
        
        # 获取最近已做的检查
        recent_tests = self._get_recent_tests(program_state)
        
        # 构建检查列表信息（包含价格和准确性）
        tests_info = []
        for test in available_tests:
            cost = MedicalConfig.TEST_COSTS[test]
            accuracy = MedicalConfig.TEST_ACCURACY.get(test, 0.7)
            affordability = "✅" if cost <= program_state.remaining_budget else "❌"
            
            tests_info.append(f"{test}: {cost}元 (准确率{accuracy:.0%}) {affordability}")
        
        # ==================== 在这里修改提示词 ====================
        prompt = f"""你是一位经验丰富的医生，正在为患者选择检查项目。

    【患者症状】
    {symptoms_text}

    【近期对话历史】
    {history_text}

    【患者剩余预算】
    {program_state.remaining_budget}元

    【检查项目列表】
    {chr(10).join(tests_info)}

    【重要说明】
    1. 必须从上述检查项目中选择
    2. 必须选择在预算范围内的检查（标记为✅的项目）
    3. 优先选择与症状最相关的检查
    4. 避免重复最近已做的检查：{recent_tests if recent_tests else "无"}
    5. 考虑检查的临床价值和必要性
    6. 💡 重要提醒：患者的理想预算可能比剩余预算少，请谨慎选择，若检查太多可考虑不检查

    【决策建议】
    - 如果当前信息已经足够诊断，可以选择"血常规"作为基础检查
    - 如果症状不典型或需要排除其他疾病，选择针对性强的检查
    - 平衡诊断需求和费用控制

    请根据患者的症状选择最合适的1项检查，直接输出检查名称（仅名称）："""
        # ==================== 修改结束 ====================
        
        try:
            response = self.api_client.chat(
                system_prompt="你是一位专业的医学专家，擅长根据症状选择恰当的检查项目",
                user_message=prompt,
                temperature=0.4  # 中等温度平衡专业性和灵活性
            )
            
            # 从响应中提取检查名称
            selected_test = self._extract_test_from_response(response, available_tests, program_state.remaining_budget)
            
            # 如果成功选择了有效的检查，返回它
            if selected_test:
                return selected_test
            else:
                # 如果AI选择失败，回退到基础检查
                return self._select_basic_test(program_state.remaining_budget)
                
        except Exception as e:
            print(f"⚠️ AI选择检查时出错: {e}")
            return self._select_basic_test(program_state.remaining_budget)
    
    def _extract_test_from_response(self, response: str, available_tests: List[str], budget: int) -> str:
        """从AI响应中提取检查名称"""
        # 清理响应
        response = response.strip()
        
        # 尝试直接匹配
        for test in available_tests:
            # 检查名称是否出现在响应中
            if test in response:
                # 验证预算
                if MedicalConfig.TEST_COSTS[test] <= budget:
                    return test
        
        # 如果直接匹配失败，尝试部分匹配
        for test in available_tests:
            test_words = test.replace("检查", "").replace("测试", "").replace("检测", "").strip()
            if test_words in response:
                if MedicalConfig.TEST_COSTS[test] <= budget:
                    return test
        
        # 检查是否有类似"血常规检查"、"心电图检查"这样的完整名称
        for test in available_tests:
            if f"{test}检查" in response or f"{test}测试" in response or f"{test}检测" in response:
                if MedicalConfig.TEST_COSTS[test] <= budget:
                    return test
        
        return ""
    
    def _select_basic_test(self, budget: int) -> str:
        """选择基础检查（当AI选择失败时使用）"""
        # 获取预算内的检查
        affordable_tests = [
            test for test, cost in MedicalConfig.TEST_COSTS.items()
            if cost <= budget
        ]
        
        if not affordable_tests:
            # 如果预算不够任何检查，返回最便宜的
            cheapest = min(MedicalConfig.TEST_COSTS.items(), key=lambda x: x[1])
            return cheapest[0]
        
        # 按价格排序，选择中等价格的检查（避免总是选择最便宜的）
        affordable_tests.sort(key=lambda x: MedicalConfig.TEST_COSTS[x])
        
        # 选择价格在中间位置的检查（增加多样性）
        if len(affordable_tests) >= 3:
            return affordable_tests[len(affordable_tests) // 2]  # 中间位置
        else:
            return affordable_tests[0]  # 第一个
    
    def _get_recent_tests(self, program_state: programState) -> List[str]:
        """获取最近已做的检查"""
        recent_tests = []
        
        # 从行动历史中查找最近的检查
        for action in reversed(program_state.actions_history[-10:]):  # 查看最近10个行动
            if action.get("type") == "检查":
                test_type = action.get("details", {}).get("test_type")
                if test_type and test_type not in recent_tests:
                    recent_tests.append(test_type)
        
        return recent_tests[-3:]  # 返回最近3个检查

    def make_diagnosis(self, full_dialogue: List, test_results: List) -> str:
        """做出最终诊断"""
        dialogue_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in full_dialogue
        ])
        
        test_text = "\n".join(test_results) if test_results else "无检查结果"

        prompt = f"""根据以下医患对话和检查结果，请做出诊断：

        【对话记录】
        {dialogue_text}

        【检查结果】
        {test_text}

        {self.historical_experience if self.historical_experience else ''}

        请输出最可能的疾病诊断："""

        diagnosis = self.api_client.chat(
            system_prompt="你是一个专业的医疗诊断专家",
            user_message=prompt,
            temperature=MedicalConfig.TEMPERATURE_DOCTOR_DIAGNOSIS
        )
        return diagnosis
    
    def learn_from_round(self, round_result: Dict, run_id: str):
        """从本轮学习并更新长期记忆"""
        self.learning_history.append(round_result)
        
        # 提取关键学习点
        key_learning = self._extract_key_learning(round_result)
        
        # 更新策略
        strategy_key = f"q{round_result['questions_asked']}_t{round_result['tests_ordered']}"
        if round_result["success"]:
            self.successful_strategies[strategy_key] = \
                self.successful_strategies.get(strategy_key, 0) + 1
        else:
            self.successful_strategies[strategy_key] = \
                self.successful_strategies.get(strategy_key, 0) - 1
        
        # 保存到长期记忆
        if MedicalConfig.ENABLE_LONG_TERM_MEMORY:
            learning_experience = {
                "success_rate": round_result["success"],
                "avg_questions": round_result["questions_asked"],
                "avg_tests": round_result["tests_ordered"],
                "cost_efficiency": round_result.get("cost_ratio", 1.0),
                "key_learning": key_learning,
                "strategy_used": strategy_key
            }
            self.memory_manager.save_learning_experience(learning_experience, run_id)
    
    def _extract_key_learning(self, round_result: Dict) -> str:
        """从回合结果中提取关键学习点"""
        if round_result["success"]:
            if round_result["questions_asked"] <= 3:
                return "少量精准提问即可确诊"
            elif round_result["tests_ordered"] > 0:
                return "合理使用检查提高诊断准确性"
            else:
                return "纯问诊也能成功诊断"
        else:
            if round_result["final_suspicion"] >= MedicalConfig.SUSPICION_THRESHOLD:
                return "患者信任管理需要改进"
            elif round_result.get("cost_ratio", 1) > 2.0:
                return "费用控制需要优化"
            else:
                return "需要提高诊断准确性"

    def get_learning_summary(self) -> str:
        """获取学习摘要"""
        if not self.learning_history:
            return "尚无学习数据"
        
        recent_rounds = self.learning_history[-5:] if len(self.learning_history) >= 5 else self.learning_history
        success_rate = sum(1 for r in recent_rounds if r["success"]) / len(recent_rounds)
        avg_questions = sum(r["questions_asked"] for r in recent_rounds) / len(recent_rounds)
        avg_tests = sum(r["tests_ordered"] for r in recent_rounds) / len(recent_rounds)
        
        return (f"近期成功率: {success_rate:.1%} | "
                f"平均问题: {avg_questions:.1f} | "
                f"平均检查: {avg_tests:.1f}")
    
    def export_learning_data(self) -> Dict:
        """导出学习数据"""
        return {
            "learning_history": self.learning_history,
            "successful_strategies": self.successful_strategies,
            "total_rounds_learned": len(self.learning_history)
        }


# ==================== 生成器 ====================

class CaseGenerator:
    """病例生成器"""

    def __init__(self, api_client: DeepSeekClient):
        self.api_client = api_client

    def generate_random_case(self) -> Dict:
        """生成随机病例"""
        disease = random.choice(MedicalConfig.DISEASE_LIBRARY)
        personality = random.choice(list(MedicalConfig.PERSONALITY_TYPES.keys()))
        personality_info = MedicalConfig.PERSONALITY_TYPES[personality]
        
        # 生成症状描述
        symptoms = self._generate_symptoms_description(disease)
        
        # 生成理想费用
        cost_range = personality_info["ideal_cost_range"]
        ideal_cost = random.randint(cost_range[0], cost_range[1])
        
        return {
            "true_disease": disease,
            "symptoms_description": symptoms,
            "personality": personality,
            "ideal_cost": ideal_cost
        }

    def _generate_symptoms_description(self, disease: str) -> str:
        """生成症状描述"""
        prompt = f"""请为{disease}患者生成一个真实的病情描述，要求：
1. 包含2-4个典型症状
2. 症状描述要自然、口语化
3. 包含一些模糊表达（如"有点"、"好像"、"说不清"）
4. 不超过80字

输出症状描述："""

        response = self.api_client.chat(
            system_prompt="你是一个真实患者，正在描述自己的病情",
            user_message=prompt,
            temperature=MedicalConfig.TEMPERATURE_CASE_GENERATION
        )
        return response.strip()


# ==================== 引擎 ====================

class MedicalDiagnosisprogram:
    """医疗诊断引擎"""

    def __init__(self, auto_mode: bool = False):
        self.api_client = DeepSeekClient()
        self.medical_system = MedicalSystem()
        self.case_generator = CaseGenerator(self.api_client)
        self.doctor = DoctorAgent(self.api_client)
        self.record_manager = RecordManager()
        self.auto_mode = auto_mode
        self.total_rounds = 0
        self.program_results = []
        self.run_id = None

    def extract_symptoms_from_complaint(self, complaint: str) -> List[str]:
        """从患者主诉中提取症状关键词"""
        symptom_keywords = [
            "头痛", "头晕", "发热", "咳嗽", "胸痛", "腹痛", "恶心", "呕吐",
            "乏力", "食欲", "多饮", "多尿", "心悸", "气短", "关节痛", "皮疹",
            "失眠", "焦虑", "抑郁", "视力模糊", "耳鸣", "鼻塞", "流涕"
        ]
        found_symptoms = []
        for symptom in symptom_keywords:
            if symptom in complaint:
                found_symptoms.append(symptom)
        return found_symptoms
    
    def print_section(self, title: str, color: str = Fore.YELLOW):
        """打印章节标题分隔符"""
        separator = "=" * 60
        print(f"\n{color}{separator}")
        print(f"{title:^60}")
        print(f"{separator}{Style.RESET_ALL}\n")
    def _doctor_decide_continue(self, program_state, patient) -> bool:
        """医生决定是否继续收集证据"""
    
    # 如果有充足预算且患者怀疑度不高，医生可能想多收集证据
        if program_state.remaining_budget > 200 and patient.suspicion_level < 0.5:
            prompt = f"""作为医生，你已收集到初步证据，但：
    - 患者怀疑度较低 ({patient.suspicion_level:.2f})
    - 还有充足预算 ({program_state.remaining_budget}元)

    你是否想再问1-2个问题或做一个检查来确认诊断？
    回答"继续问诊"或"停止问诊"："""
            
            try:
                response = self.doctor.api_client.chat(
                    system_prompt="你是谨慎的医生，会权衡证据充分性和患者感受",
                    user_message=prompt,
                    temperature=0.4
                ).strip()
                
                return "继续问诊" in response
            except:
                # 默认：如果预算充足且患者不怀疑，继续
                return program_state.remaining_budget > 150 and patient.suspicion_level < 0.4
        else:
            # 预算紧张或患者怀疑度高时，立即停止
            return False

    def print_info(self, message: str, color: str = Fore.WHITE):
        """打印信息"""
        print(f"{color}{message}{Style.RESET_ALL}")

    def play_round(self) -> Dict:
        """进行一轮诊断"""
        self.total_rounds += 1
        self.print_section(f"🩺 第 {self.total_rounds} 位患者就诊", Fore.CYAN)

        # 生成病例和患者
        case_info = self.case_generator.generate_random_case()
        patient = PatientAgent(self.api_client, case_info)
        program_state = programState()
        program_state.current_round = self.total_rounds

        # 显示病例信息
        self.print_info(f"【患者个性】{case_info['personality']}", Fore.MAGENTA)
        self.print_info(f"【理想费用】{case_info['ideal_cost']}元", Fore.MAGENTA)
        self.print_info(f"【真实病情】{case_info['true_disease']}", Fore.GREEN)
        
        # 患者主诉
        self.print_info("\n患者主诉:", Fore.YELLOW)
        initial_complaint = patient.get_initial_complaint()
        self.print_info(f"患者: {initial_complaint}", Fore.WHITE)
        patient_symptoms = self.extract_symptoms_from_complaint(initial_complaint)
        program_state.patient_symptoms = patient_symptoms
        program_state.dialogue_history = patient.dialogue_history.copy()

        # 主循环
        while not program_state.is_round_over(self.doctor):  # 传入doctor参数
            self.print_info(f"\n{program_state.get_status_summary()}", Fore.CYAN)
            
            # 如果证据已充分但还没跳出循环，直接结束
            if program_state.evidence_sufficient:
                self.print_info("🧠 医生认为证据已充分，停止问诊", Fore.GREEN)
                break
                
            # 医生选择行动
            action = self.doctor.choose_action(program_state, patient)
            
            if action == "询问病情":
                self._handle_questioning(program_state, patient, program_state.dialogue_history)
            else:
                self._handle_test_ordering(program_state, patient, program_state.dialogue_history, program_state.test_results)
            
            # 每次行动后，医生重新评估证据是否充分
            if program_state.questions_asked >= 4 or program_state.tests_ordered >= 1:
                # 医生评估
                is_sufficient = self.doctor.is_evidence_sufficient(
                    program_state.dialogue_history,
                    program_state.test_results,
                    program_state.current_round,
                    patient.suspicion_level
                )
                
                if is_sufficient and not program_state.evidence_sufficient:
                    program_state.evidence_sufficient = True
                    self.print_info("🧠 医生认为当前证据已足够诊断", Fore.GREEN)
                    # 可以选择继续问诊或立即结束
                    # 这里让医生决定是否继续
                    continue_action = self._doctor_decide_continue(program_state, patient)
                    if not continue_action:
                        break
        
            # if not self.auto_mode and not program_state.is_round_over(self.doctor):
            #     input("按回车继续...")

        # 最终诊断和评估
        round_result = self._evaluate_round(program_state, patient, case_info, program_state.dialogue_history, program_state.test_results)
        
        # 保存本轮记录
        if MedicalConfig.SAVE_RECORDS:
            round_data = self._prepare_round_data(program_state, patient, case_info, round_result)
            round_file = self.record_manager.save_round_log(round_data, self.total_rounds)
            self.print_info(f"💾 本轮记录已保存: {round_file}", Fore.GREEN)
        
        return round_result

    def _handle_questioning(self, program_state: programState, patient: PatientAgent, 
                          dialogue_history: List):
        """处理询问病情"""
        self.print_info("\n💬 医生询问病情", Fore.BLUE)
        
        question = self.doctor.generate_question(dialogue_history)
        self.print_info(f"医生: {question}", Fore.BLUE)
        
        response = patient.respond_to_question(question)
        self.print_info(f"患者: {response}", Fore.WHITE)
        
        program_state.add_question()
        program_state.record_action("询问", {"question": question, "response": response})
        
        dialogue_history.extend([
            {"role": "doctor", "content": question},
            {"role": "patient", "content": response}
        ])

    def _handle_test_ordering(self, program_state: programState, patient: PatientAgent,
                            dialogue_history: List, test_results: List):
        """处理检查要求"""
        self.print_info("\n🔬 医生要求检查", Fore.GREEN)
        
        test_type = self.doctor.select_test_type(program_state, program_state.patient_symptoms, dialogue_history)
        if not test_type:
            test_type = "血常规"  # 终极后备
        self.print_info(f"医生: 建议进行{test_type}检查", Fore.GREEN)
        
        test_result = self.medical_system.perform_test(test_type, patient.true_condition)
        self.print_info(f"检查结果: {test_result['result']}", Fore.WHITE)
        self.print_info(f"检查费用: {test_result['cost']}元", Fore.YELLOW)
        
        program_state.add_test(test_result['cost'])
        program_state.record_action("检查", {
            "test_type": test_type, 
            "result": test_result['result'],
            "cost": test_result['cost'],
            "accurate": test_result['accurate']
        })
        
        test_results.append(f"{test_type}: {test_result['result']}")
        
        dialogue_history.append({
            "role": "system", 
            "content": f"进行了{test_type}检查，结果: {test_result['result']}"
        })
    def _get_round_end_reason(self, program_state: programState) -> str:
        """获取回合结束原因"""
        if program_state.patient_suspicion >= MedicalConfig.SUSPICION_THRESHOLD:
            return "患者怀疑度过高"
        elif program_state.remaining_budget <= 0:
            return "预算耗尽"
        elif program_state.questions_asked >= MedicalConfig.MAX_QUESTIONS_PER_ROUND:
            return "问题数达到上限"
        elif program_state.evidence_sufficient:
            return "医生认为证据充分"
        else:
            return "未知原因"

    def _evaluate_round(self, program_state: programState, patient: PatientAgent, 
                       case_info: Dict, dialogue_history: List, test_results: List) -> Dict:
        """评估本轮结果"""
        self.print_section("📊 回合评估", Fore.MAGENTA)

        # 失败条件检查
        failure_reasons = []
        if patient.suspicion_level >= MedicalConfig.SUSPICION_THRESHOLD:
            failure_reasons.append("患者信任丧失")
        if program_state.remaining_budget < 0:
            failure_reasons.append("预算耗尽")
        if program_state.questions_asked >= MedicalConfig.MAX_QUESTIONS_PER_ROUND:
            failure_reasons.append("问题数超限")

        # 最终诊断
        self.print_info("🤔 医生思考最终诊断...", Fore.CYAN)
        diagnosis = self.doctor.make_diagnosis(dialogue_history, test_results)
        self.print_info(f"医生诊断: {diagnosis}", Fore.CYAN)

        # 判断诊断准确性
        diagnosis_correct = case_info["true_disease"] in diagnosis
        cost_ratio = program_state.total_cost / case_info["ideal_cost"]

        # 综合评估
        success = (diagnosis_correct and 
                  not failure_reasons and 
                  cost_ratio <= 2.0)  # 费用不超过理想费用2倍

        if success:
            self.print_info("✅ 问诊成功！", Fore.GREEN)
        else:
            self.print_info("❌ 问诊失败", Fore.RED)
            if failure_reasons:
                self.print_info(f"失败原因: {', '.join(failure_reasons)}", Fore.RED)
            if not diagnosis_correct:
                self.print_info("诊断不正确", Fore.RED)
            if cost_ratio > 2.0:
                self.print_info(f"费用超标 (实际: {program_state.total_cost}元, 理想: {case_info['ideal_cost']}元)", Fore.RED)

        round_result = {
            "round": self.total_rounds,
            "success": success,
            "true_disease": case_info["true_disease"],
            "diagnosis": diagnosis,
            "diagnosis_correct": diagnosis_correct,
            "questions_asked": program_state.questions_asked,
            "tests_ordered": program_state.tests_ordered,
            "total_cost": program_state.total_cost,
            "ideal_cost": case_info["ideal_cost"],
            "final_suspicion": patient.suspicion_level,
            "failure_reasons": failure_reasons,
            "cost_ratio": cost_ratio,
            "evidence_sufficient": program_state.evidence_sufficient,  # 新增
            "round_end_reason": self._get_round_end_reason(program_state)
        }

        # 医生学习
        self.doctor.learn_from_round(round_result, self.run_id)

        # 显示学习进度
        learning_summary = self.doctor.get_learning_summary()
        self.print_info(f"\n📈 学习进度: {learning_summary}", Fore.CYAN)

        return round_result

    def _prepare_round_data(self, program_state: programState, patient: PatientAgent, 
                           case_info: Dict, round_result: Dict) -> Dict:
        """准备本轮数据用于保存"""
        return {
            "round_info": {
                "round_number": self.total_rounds,
                "start_time": program_state.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "result": round_result
            },
            "patient_info": patient.export_to_dict(),
            "case_info": case_info,
            "program_state": program_state.export_to_dict(),
            "doctor_learning": self.doctor.export_learning_data()
        }

    def run_program(self, total_rounds: int = 5):
        """运行完整程序"""
        self.print_section("🏥 AI医患诊断开始", Fore.CYAN)
        self.print_info("规则:", Fore.YELLOW)
        self.print_info("• 医生要通过询问和检查诊断疾病", Fore.WHITE)
        self.print_info("• 患者描述可能模糊或不准确", Fore.WHITE)
        self.print_info("• 检查准确但增加费用和患者怀疑", Fore.WHITE)
        self.print_info("• 需要在信任、费用、准确性间平衡", Fore.WHITE)

        MedicalConfig.validate()

        self.program_results = []
        program_start_time = datetime.now()
        
        for round_num in range(total_rounds):
            result = self.play_round()
            self.program_results.append(result)
            
            if round_num < total_rounds - 1:
                if not self.auto_mode:
                    input("\n按回车继续下一位患者...")
                else:
                    print("\n" + "="*60)
                    time.sleep(2)

        # 保存完整记录
        if MedicalConfig.SAVE_RECORDS:
            self.run_id = self._save_complete_program_record(program_start_time, total_rounds)

        # 最终报告
        self._show_final_report()

    def _save_complete_program_record(self, start_time: datetime, total_rounds: int) -> str:
        """保存完整记录"""
        program_data = {
            "program_info": {
                "total_rounds": total_rounds,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_duration_seconds": (datetime.now() - start_time).total_seconds()
            },
            "program_results": self.program_results,
            "doctor_final_learning": self.doctor.export_learning_data(),
            "performance_summary": self._calculate_performance_summary()
        }
        
        run_id = self.record_manager.save_program_record(program_data)
        self.print_info(f"💾 完整记录已保存，ID: {run_id}", Fore.GREEN)
        return run_id

    def _calculate_performance_summary(self) -> Dict:
        """计算性能摘要"""
        total_success = sum(1 for r in self.program_results if r["success"])
        success_rate = total_success / len(self.program_results)
        
        avg_questions = sum(r["questions_asked"] for r in self.program_results) / len(self.program_results)
        avg_tests = sum(r["tests_ordered"] for r in self.program_results) / len(self.program_results)
        avg_cost = sum(r["total_cost"] for r in self.program_results) / len(self.program_results)
        avg_cost_ratio = sum(r["cost_ratio"] for r in self.program_results) / len(self.program_results)

        return {
            "success_rate": success_rate,
            "avg_questions": avg_questions,
            "avg_tests": avg_tests,
            "avg_cost": avg_cost,
            "avg_cost_ratio": avg_cost_ratio,
            "total_rounds": len(self.program_results)
        }

    def _show_final_report(self):
        """显示最终报告"""
        self.print_section("🎓 最终报告", Fore.MAGENTA)
        
        performance = self._calculate_performance_summary()
        
        self.print_info(f"总回合数: {performance['total_rounds']}", Fore.CYAN)
        self.print_info(f"成功率: {performance['success_rate']:.1%}", 
                       Fore.GREEN if performance['success_rate'] > 0.5 else Fore.RED)
        self.print_info(f"平均问题数: {performance['avg_questions']:.1f}", Fore.CYAN)
        self.print_info(f"平均检查数: {performance['avg_tests']:.1f}", Fore.CYAN)
        self.print_info(f"平均费用: {performance['avg_cost']:.1f}元", Fore.CYAN)
        self.print_info(f"平均费用比率: {performance['avg_cost_ratio']:.1f}", 
                       Fore.GREEN if performance['avg_cost_ratio'] <= 1.5 else Fore.YELLOW if performance['avg_cost_ratio'] <= 2.0 else Fore.RED)

        # 显示医生学习总结
        learning_summary = self.doctor.get_learning_summary()
        self.print_info(f"\n医生学习总结: {learning_summary}", Fore.CYAN)
        
        # 显示记录保存信息
        if self.run_id:
            self.print_info(f"\n📁 记录已保存到: {MedicalConfig.RECORDS_DIRC}/", Fore.GREEN)
            self.print_info(f"📁 回合日志已保存到: {MedicalConfig.ROUND_LOGS_DIR}/", Fore.GREEN)
            self.print_info(f"📁 医生记忆已保存到: {MedicalConfig.DOCTOR_MEMORY_DIR}/", Fore.GREEN)


# ==================== 主程序 ====================

def print_banner():
    """打印欢迎横幅"""
    banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗
{Fore.CYAN}║                                                              ║
{Fore.CYAN}║              🏥 AI 医患诊断学习                               ║
{Fore.CYAN}║                                                              ║
{Fore.CYAN}║        医生智能体 vs 患者智能体 - 多轮学习进化               ║
{Fore.CYAN}║                    带完整记录系统                           ║
{Fore.CYAN}║                                                              ║
{Fore.CYAN}╚══════════════════════════════════════════════════════════════╝
{Style.RESET_ALL}"""
    print(banner)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AI医患诊断')
    parser.add_argument('--auto', action='store_true', help='自动模式（无需交互）')
    parser.add_argument('--rounds', type=int, default=5, help='回合数')
    args = parser.parse_args()

    try:
        print_banner()
        program = MedicalDiagnosisprogram(auto_mode=args.auto)
        program.run_program(total_rounds=args.rounds)
        
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}程序被用户中断{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}❌ 程序错误: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    main()
