"""
AI Doctor-Patient Diagnostic System - with complete records and long-term learning mechanism
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

# Initialize colorama
init(autoreset=True)

# Load environment variables
load_dotenv()


# ==================== Configuration Class ====================

class MedicalConfig:
    """Medical configuration class"""
    
    # ==================== API Configuration ====================
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    MODEL_NAME = "deepseek-chat"

    # ==================== Basic Configuration ====================
    MAX_QUESTIONS_PER_ROUND = 12  # Maximum questions per round
    INITIAL_BUDGET = 500  # Initial budget
    SUSPICION_THRESHOLD = 0.8  # Suspicion threshold
    
    # ==================== Display Configuration ====================
    SHOW_AI_THINKING = True  # Show AI thinking process
    SHOW_DETAILED_LOGS = True  # Show detailed logs
    
    # ==================== Record Configuration ====================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_RECORDS = True  # Save records
    RECORDS_DIRC = os.path.join(BASE_DIR, "medical_records")
    DOCTOR_MEMORY_DIR = os.path.join(BASE_DIR, "doctor_memory") 
    ROUND_LOGS_DIR = os.path.join(BASE_DIR, "round_logs")
    ENABLE_LONG_TERM_MEMORY = True  # Enable long-term memory
    MAX_HISTORY = 10  # Save last 10 records
    
    # ==================== Cost Configuration ====================
    QUESTION_COST = 0  # Questions are free
    TEST_COSTS = {
    "Blood Test": 80,
    "Urine Test": 60, 
    "ECG": 120,
    "Chest X-ray": 150,
    "CT Scan": 300,
    "MRI": 500,
    "Ultrasound": 200,
    "Gastroscopy": 400,
    # Additional items
    "Liver Function Test": 90,
    "Kidney Function Test": 85,
    "Blood Glucose Test": 50,
    "Lipid Panel": 110,
    "Bone Density Test": 180,
    "Endoscopy": 350,
    "Biopsy": 250,
    "EEG": 160,
    "Pulmonary Function Test": 130,
    "Skin Allergy Test": 95
    }
    
    TEST_ACCURACY = {
    "Blood Test": 0.7,
    "Urine Test": 0.65,
    "ECG": 0.8,
    "Chest X-ray": 0.75,
    "CT Scan": 0.9,
    "MRI": 0.95,
    "Ultrasound": 0.85,
    "Gastroscopy": 0.88,
    # Additional items
    "Liver Function Test": 0.72,
    "Kidney Function Test": 0.68,
    "Blood Glucose Test": 0.95,
    "Lipid Panel": 0.82,
    "Bone Density Test": 0.88,
    "Endoscopy": 0.92,
    "Biopsy": 0.96,
    "EEG": 0.78,
    "Pulmonary Function Test": 0.85,
    "Skin Allergy Test": 0.9
    }


    # ==================== AI Parameter Configuration ====================
    # Temperature parameters - different scenarios use different temperatures
    TEMPERATURE_PATIENT_RESPONSE = 0.9    # Patient response - high temperature increases diversity
    TEMPERATURE_DOCTOR_QUESTION = 0.7     # Doctor questions - medium temperature balances professionalism and flexibility
    TEMPERATURE_DOCTOR_DIAGNOSIS = 0.3    # Doctor diagnosis - low temperature ensures accuracy
    TEMPERATURE_CASE_GENERATION = 0.6     # Case generation - medium temperature ensures realism
    
    MAX_TOKENS = 800

    # ==================== Disease Library ====================
    DISEASE_LIBRARY = [
    "Migraine", "Gastritis", "Allergic Rhinitis", "Common Cold", "Hypertension", 
    "Diabetes", "Asthma", "Arthritis", "Skin Disease", "Insomnia",
    # Additional diseases
    "Pneumonia", "Bronchitis", "Gastric Ulcer", "Kidney Stones", "Cholecystitis",
    "Myocarditis", "Concussion", "Lumbar Disc Herniation", "Osteoporosis", "Anemia",
    "Hyperthyroidism", "Gout", "Hepatitis", "Irritable Bowel Syndrome", "Depression",
    "Anxiety Disorder", "Cataracts", "Glaucoma", "Otitis Media", "Sinusitis"
    ]

    # ==================== Patient Personality Types ====================
    PERSONALITY_TYPES = {
    "Cautious": {"suspicion_gain": 0.15, "cost_sensitivity": 0.8, "ideal_cost_range": (80, 150)},
    "Casual": {"suspicion_gain": 0.08, "cost_sensitivity": 0.4, "ideal_cost_range": (120, 200)},
    "Hypochondriac": {"suspicion_gain": 0.25, "cost_sensitivity": 0.3, "ideal_cost_range": (150, 250)},
    "Frugal": {"suspicion_gain": 0.12, "cost_sensitivity": 0.9, "ideal_cost_range": (50, 100)},
    # Additional personality types
    "Impatient": {"suspicion_gain": 0.20, "cost_sensitivity": 0.5, "ideal_cost_range": (100, 180)},
    "Dependent": {"suspicion_gain": 0.05, "cost_sensitivity": 0.6, "ideal_cost_range": (200, 300)},
    "Rational": {"suspicion_gain": 0.10, "cost_sensitivity": 0.7, "ideal_cost_range": (150, 220)},
    "Suspicious": {"suspicion_gain": 0.30, "cost_sensitivity": 0.4, "ideal_cost_range": (80, 120)}
    }

    # ==================== Misunderstanding Triggers ====================
    MISUNDERSTANDING_TRIGGERS = {
    "eating": {"threshold": 0.4, "misunderstanding": "Doesn't consider eating a few hours ago as 'just ate'"},
    "drinking": {"threshold": 0.3, "misunderstanding": "Doesn't consider beer as 'drinking alcohol'"},
    "exercise": {"threshold": 0.5, "misunderstanding": "Doesn't consider walking as 'exercise'"},
    "sleep": {"threshold": 0.4, "misunderstanding": "Counts napping as 'sleeping'"},
    "pain": {"threshold": 0.6, "misunderstanding": "Can't distinguish between soreness and sharp pain"},
    # Additional triggers
    "nausea": {"threshold": 0.35, "misunderstanding": "Describes stomach discomfort as nausea"},
    "dizziness": {"threshold": 0.45, "misunderstanding": "Can't distinguish between dizziness and vertigo"},
    "fever": {"threshold": 0.3, "misunderstanding": "Considers normal temperature fluctuations as fever"},
    "cough": {"threshold": 0.4, "misunderstanding": "Counts throat clearing as coughing"},
    "fatigue": {"threshold": 0.5, "misunderstanding": "Describes normal tiredness as pathological fatigue"},
    "appetite": {"threshold": 0.35, "misunderstanding": "Attributes bad mood to lack of appetite"},
    "medication": {"threshold": 0.4, "misunderstanding": "Forgets medication or remembers wrong dosage"},
    "time": {"threshold": 0.6, "misunderstanding": "Remembers wrong symptom onset time"},
    "frequency": {"threshold": 0.55, "misunderstanding": "Exaggerates or minimizes symptom frequency"},
    "location": {"threshold": 0.5, "misunderstanding": "Inaccurately describes pain location"}
    }

    @classmethod
    def validate(cls):
        """Validate configuration effectiveness"""
        if not cls.DEEPSEEK_API_KEY:
            raise ValueError(
                "❌ Error: DEEPSEEK_API_KEY not found!\n"
                "Please set DEEPSEEK_API_KEY=your_api_key in .env file\n"
                "Or set environment variable: export DEEPSEEK_API_KEY=your_api_key"
            )
        
        # Create record directories
        if cls.SAVE_RECORDS:
            os.makedirs(cls.RECORDS_DIRC, exist_ok=True)
            os.makedirs(cls.DOCTOR_MEMORY_DIR, exist_ok=True)
            os.makedirs(cls.ROUND_LOGS_DIR, exist_ok=True)
            
        print("✅ Medical configuration validated successfully")
        return True


# ==================== Memory Management System ====================

class MemoryManager:
    """Memory Manager - handles doctor's long-term learning memory"""
    
    def __init__(self):
        self.memory_dir = MedicalConfig.DOCTOR_MEMORY_DIR
        self.memory_file = os.path.join(self.memory_dir, "doctor_memory.json")
        os.makedirs(self.memory_dir, exist_ok=True)
    
    def save_learning_experience(self, experience: Dict, run_id: str):
        """Save learning experience to long-term memory"""
        memories = self._load_memory()
        
        memories.append({
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "experience": experience
        })
        
        # Limit memory count
        if len(memories) > MedicalConfig.MAX_HISTORY:
            memories = memories[-MedicalConfig.MAX_HISTORY:]
            
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(memories, f, ensure_ascii=False, indent=2)
    
    def load_learning_experience(self) -> str:
        """Load long-term learning experience"""
        memories = self._load_memory()
        
        if not memories:
            return "No historical learning experience available"
        
        experience_parts = []
        experience_parts.append("【Doctor's Historical Learning Experience】")
        experience_parts.append(f"(Based on experience from recent {len(memories)} sessions)")
        
        for i, memory in enumerate(memories[-5:], 1):  # Show recent 5 sessions
            exp = memory['experience']
            exp_summary = f"Diagnosis {i}: Success rate {exp.get('success_rate', 0):.1%}, Avg questions {exp.get('avg_questions', 0):.1f}, Key learning: {exp.get('key_learning', '')}"
            experience_parts.append(exp_summary)
        
        return "\n".join(experience_parts)
    
    def _load_memory(self) -> list:
        """Load memory file"""
        if not os.path.exists(self.memory_file):
            return []
        
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []


# ==================== Record System ====================

class RecordManager:
    """Record Manager - handles records and round logs"""
    
    def __init__(self):
        self.RECORDS_DIRC = MedicalConfig.RECORDS_DIRC
        self.round_logs_dir = MedicalConfig.ROUND_LOGS_DIR
    
    def save_program_record(self, program_data: Dict) -> str:
        """Save complete program record"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"program_{timestamp}.json"
        filepath = os.path.join(self.RECORDS_DIRC, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(program_data, f, ensure_ascii=False, indent=2)
        
        return timestamp
    
    def save_round_log(self, round_data: Dict, round_number: int) -> str:
        """Save detailed single round log"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"round_{round_number}_{timestamp}.json"
        filepath = os.path.join(self.round_logs_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(round_data, f, ensure_ascii=False, indent=2)
        
        return filepath


# ==================== API Client ====================

class DeepSeekClient:
    """DeepSeek API Client Class"""

    def __init__(self):
        """Initialize DeepSeek client"""
        self.client = OpenAI(
            api_key=MedicalConfig.DEEPSEEK_API_KEY,
            base_url=MedicalConfig.DEEPSEEK_BASE_URL
        )
        self.model = MedicalConfig.MODEL_NAME
        self.max_tokens = MedicalConfig.MAX_TOKENS

    def chat(self, system_prompt: str, user_message: str, temperature: float = 0.7) -> str:
        """Send chat request to DeepSeek API"""
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
                print(f"⏱️  API response time: {elapsed_time:.2f}s")
            return reply

        except Exception as e:
            error_msg = f"❌ DeepSeek API call failed: {str(e)}"
            print(error_msg)
            # Return fallback response
            return "I need more information to assess your condition."


# ==================== Medical System ====================

class MedicalSystem:
    """Medical System - handles test execution and cost calculation"""
    TEST_DISEASE_RELEVANCE = {
    # ==================== Blood/Biochemical Tests ====================
    "Blood Glucose Test": {
        "Diabetes": 0.95,        # Direct diagnostic basis
        "Hypertension": 0.25,    # May accompany glucose metabolism abnormalities
        "Hyperthyroidism": 0.20, # May affect blood glucose
        "Common Cold": 0.05,     # Basically unrelated
        "Gastritis": 0.10,       # Stomach issues may affect eating, indirectly related
        "Pneumonia": 0.10,       # Infection may cause stress hyperglycemia
    },
    
    "Glycated Hemoglobin": {
        "Diabetes": 0.90,        # Reflects long-term blood glucose control
        "Anemia": 0.40,          # Affects HbA1c measurement
        "Kidney Disease": 0.35,  # Kidney function affects HbA1c
    },
    
    "Blood Test": {
        "Infectious Diseases": 0.80,    # White blood cell count is infection marker
        "Pneumonia": 0.75,
        "Common Cold": 0.65,
        "Bronchitis": 0.70,
        "Anemia": 0.85,          # Hemoglobin is direct indicator
        "Leukemia": 0.90,
        "Gastritis": 0.40,       # May accompany chronic blood loss
        "Diabetes": 0.30,        # May complicate with infection
        "Migraine": 0.10,        # Basically unrelated
    },
    
    "Liver Function Test": {
        "Hepatitis": 0.90,
        "Cirrhosis": 0.85,
        "Cholecystitis": 0.60,
        "Drug-induced Liver Injury": 0.80,
        "Diabetes": 0.25,        # May complicate with fatty liver
        "Hypertension": 0.15,
    },
    
    "Kidney Function Test": {
        "Kidney Disease": 0.90,
        "Kidney Stones": 0.70,
        "Hypertension": 0.60,    # Hypertensive nephropathy
        "Diabetes": 0.65,        # Diabetic nephropathy
        "Gout": 0.50,            # May affect kidney function
    },
    
    "Lipid Panel": {
        "Hypertension": 0.60,    # Often accompanied by lipid abnormalities
        "Diabetes": 0.65,        # Often accompanied by lipid abnormalities
        "Heart Disease": 0.70,   # Coronary heart disease risk factor
        "Arteriosclerosis": 0.75,
    },
    
    # ==================== Imaging Tests ====================
    "Chest X-ray": {
        "Pneumonia": 0.85,       # Shows lung infiltrates
        "Tuberculosis": 0.80,    # Shows tuberculosis lesions
        "Bronchitis": 0.50,      # May only show increased markings
        "Heart Disease": 0.65,   # Shows enlarged heart shadow
        "Lung Cancer": 0.70,
        "Fracture": 0.95,        # Fracture directly visible
        "Gastritis": 0.05,       # Basically can't see stomach
        "Diabetes": 0.01,        # Completely unrelated
    },
    
    "CT Scan": {
        "Pneumonia": 0.90,       # More sensitive than X-ray
        "Concussion": 0.70,      # Excludes intracranial hemorrhage
        "Fracture": 0.95,
        "Brain Tumor": 0.85,
        "Lumbar Disc Herniation": 0.90,
        "Kidney Stones": 0.95,   # Urinary tract stones
        "Gastritis": 0.30,       # Can show gastric wall thickening
        "Heart Disease": 0.60,   # Coronary CT
    },
    
    "MRI": {
        "Concussion": 0.75,      # More sensitive to brain tissue than CT
        "Brain Tumor": 0.95,
        "Lumbar Disc Herniation": 0.95,
        "Arthritis": 0.85,       # Joint soft tissue
        "Myocarditis": 0.80,     # Cardiac MRI
        "Pneumonia": 0.60,       # Usable but not preferred
    },
    
    "Ultrasound": {
        "Cholecystitis": 0.90,   # Gallbladder wall thickening, stones
        "Kidney Stones": 0.85,
        "Cirrhosis": 0.80,       # Liver morphology
        "Hyperthyroidism": 0.75, # Thyroid size, blood flow
        "Heart Disease": 0.70,   # Cardiac ultrasound
        "Pneumonia": 0.40,       # Pleural effusion visible
        "Gastritis": 0.30,       # Can exclude other abdominal diseases
    },
    
    # ==================== Cardiac Tests ====================
    "ECG": {
        "Heart Disease": 0.90,   # Arrhythmia, myocardial ischemia
        "Myocarditis": 0.85,
        "Hypertension": 0.60,    # Left ventricular hypertrophy
        "Hyperthyroidism": 0.50, # May cause tachycardia
        "Diabetes": 0.20,        # May complicate with coronary heart disease
        "Pneumonia": 0.25,       # May have secondary cardiac effects
        "Gastritis": 0.05,       # Basically unrelated
        "Migraine": 0.05,
    },
    
    "Holter Monitor": {
        "Heart Disease": 0.95,   # Captures paroxysmal arrhythmia
        "Syncope": 0.85,         # Cardiac syncope
        "Palpitations": 0.90,
        "Myocarditis": 0.80,
    },
    
    # ==================== Endoscopic Tests ====================
    "Gastroscopy": {
        "Gastritis": 0.95,       # Direct observation of gastric mucosa
        "Gastric Ulcer": 0.90,
        "Gastric Cancer": 0.85,  # Can biopsy
        "Esophagitis": 0.80,
        "Diabetes": 0.15,        # May cause gastroparesis, but not preferred
        "Hepatitis": 0.05,       # Basically unrelated
    },
    
    "Colonoscopy": {
        "Enteritis": 0.90,
        "Colon Cancer": 0.95,
        "Irritable Bowel Syndrome": 0.30,  # Exclusion diagnosis
        "Gastritis": 0.10,       # Different location
    },
    
    # ==================== Special Tests ====================
    "Pulmonary Function Test": {
        "Asthma": 0.95,          # Obstructive ventilation dysfunction
        "Bronchitis": 0.85,
        "Pneumonia": 0.50,       # Restrictive possibility
        "Heart Disease": 0.30,   # Cardiac insufficiency effects
        "Diabetes": 0.10,
    },
    
    "Bone Density Test": {
        "Osteoporosis": 0.95,    # Direct bone density measurement
        "Fracture": 0.60,        # Assesses fracture risk
        "Arthritis": 0.40,
        "Hyperthyroidism": 0.50, # May cause bone metabolism abnormalities
    },
    
    "EEG": {
        "Epilepsy": 0.90,
        "Encephalitis": 0.75,
        "Migraine": 0.40,        # Sometimes for exclusion diagnosis
        "Concussion": 0.30,
        "Insomnia": 0.50,        # Sleep EEG
    },
    
    "Allergy Test": {
        "Allergic Rhinitis": 0.95,
        "Asthma": 0.85,          # Allergic asthma
        "Skin Disease": 0.80,    # Allergic dermatitis
        "Food Allergy": 0.90,
    },
}

    def __init__(self):
        self.test_costs = MedicalConfig.TEST_COSTS
        self.test_accuracy = MedicalConfig.TEST_ACCURACY

    def perform_test(self, test_name: str, true_condition: str) -> Dict:
        """Execute test and return result"""
        cost = self.test_costs[test_name]
        base_accuracy = self.test_accuracy[test_name]
        
        # Get test relevance to the disease
        relevance = self.TEST_DISEASE_RELEVANCE.get(test_name, {}).get(true_condition, 0.1)
        
        # Final accuracy = base accuracy × relevance
        final_accuracy = base_accuracy * relevance
        
        # Determine test result
        if random.random() < final_accuracy:
            # ✅ True positive: test correctly discovered disease
            return {
                "result": self._get_positive_result(test_name, true_condition),
                "cost": cost,
                "accurate": True,
                "relevance": relevance,  # New: record relevance
                "result_type": "true_positive"
            }
        else:
            # False negative or normal result
            if relevance < 0.3:
                # 🔍 Low relevance test: return normal result (unlikely positive anyway)
                return {
                    "result": self._get_normal_result(test_name),
                    "cost": cost,
                    "accurate": True,  # This is actually "true negative"
                    "relevance": relevance,
                    "result_type": "true_negative"  # True negative
                }
            else:
                # ❌ False negative: relevant test but missed diagnosis
                return {
                    "result": self._get_false_negative_result(test_name, true_condition),
                    "cost": cost,
                    "accurate": False,
                    "relevance": relevance,
                    "result_type": "false_negative"  # False negative
                }
    
    def _get_positive_result(self, test_name: str, disease: str) -> str:
        """Generate positive result description"""
        templates = {
            "Blood Glucose Test": f"Blood glucose test shows significantly elevated levels, meeting {disease} diagnostic criteria",
            "ECG": f"ECG shows abnormal waveform, suggesting {disease} possibility",
            "Chest X-ray": f"Chest X-ray shows lung shadowing, consistent with {disease} presentation",
            "Blood Test": f"Blood test shows multiple abnormal indicators, supporting {disease} diagnosis"
        }
        return templates.get(test_name, f"{test_name} shows abnormalities related to {disease}")
    
    def _get_false_negative_result(self, test_name: str, disease: str) -> str:
        """Generate false negative result description"""
        false_negatives = {
            "Diabetes": {
                "Blood Glucose Test": "Blood glucose levels at upper normal limit, recommend recheck",
                "Blood Test": "Blood test shows no significant abnormalities"
            },
            "Pneumonia": {
                "Chest X-ray": "Chest X-ray shows no obvious lung shadowing",
                "Blood Test": "White blood cell count mildly elevated, non-specific"
            },
            # ... Other diseases' false negative descriptions
        }
        
        return false_negatives.get(disease, {}).get(
            test_name, 
            f"{test_name} results within normal range"
        )
    
    def _get_normal_result(self, test_name: str) -> str:
        """Generate normal result description (for low relevance tests)"""
        normal_results = {
            "ECG": "ECG shows normal sinus rhythm",
            "Blood Glucose Test": "Blood glucose levels within normal range",
            "Chest X-ray": "Chest X-ray shows no significant abnormalities",
            "Blood Test": "All blood test indicators within normal range"
        }
        return normal_results.get(test_name, f"{test_name} shows no abnormalities")

    def get_available_tests(self) -> List[str]:
        """Get available test items"""
        return list(self.test_costs.keys())


# ==================== State Management ====================

class ProgramState:
    """Program State Management Class"""

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

    def record_action(self, action_type: str, details: Dict):
        """Record action history"""
        action = {
            "round": self.current_round,
            "type": action_type,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.actions_history.append(action)

    def add_question(self):
        """Increase question count"""
        self.questions_asked += 1
        self.patient_suspicion += 0.1  # Each question increases suspicion value

    def add_test(self, cost: int):
        """Increase test count and cost"""
        self.tests_ordered += 1
        self.total_cost += cost
        self.remaining_budget -= cost
        self.patient_suspicion += 0.15  

    def is_round_over(self) -> bool:
        """Check if round is over"""
        return (self.patient_suspicion >= MedicalConfig.SUSPICION_THRESHOLD or
                self.remaining_budget <= 0 or
                self.questions_asked >= MedicalConfig.MAX_QUESTIONS_PER_ROUND)

    def get_status_summary(self) -> str:
        """Get status summary"""
        return (f"Current Round: {self.current_round} | "
                f"Questions Asked: {self.questions_asked} | "
                f"Tests Ordered: {self.tests_ordered} | "
                f"Total Cost: {self.total_cost} | "
                f"Remaining Budget: {self.remaining_budget} | "
                f"Patient Suspicion: {self.patient_suspicion:.2f}")
    
    def export_to_dict(self) -> Dict:
        """Export state to dictionary"""
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


# ==================== Agent Classes ====================

class PatientAgent:
    """Patient Agent"""

    def __init__(self, api_client: DeepSeekClient, case_info: Dict):
        self.api_client = api_client
        self.true_condition = case_info["true_disease"]
        self.symptoms_description = case_info["symptoms_description"]
        self.personality = case_info["personality"]
        self.ideal_cost = case_info["ideal_cost"]
        self.suspicion_level = 0.0
        self.dialogue_history = []

    def respond_to_question(self, question: str) -> str:
        """Answer doctor's question (may be inaccurate)"""
        # Increase suspicion value
        suspicion_gain = MedicalConfig.PERSONALITY_TYPES[self.personality]["suspicion_gain"]
        self.suspicion_level += suspicion_gain

        # Determine if misunderstanding occurs
        if self._should_misunderstand(question):
            return self._generate_misunderstanding_response(question)
        else:
            return self._generate_truthful_response(question)

    def _should_misunderstand(self, question: str) -> bool:
        """Determine if misunderstanding of question occurs"""
        for trigger, info in MedicalConfig.MISUNDERSTANDING_TRIGGERS.items():
            if trigger in question and random.random() < info["threshold"]:
                return True
        return False

    def _generate_misunderstanding_response(self, question: str) -> str:
        """Generate misunderstanding response"""
        prompt = f"""You are a patient, now the doctor asks: "{question}"

Your real condition: {self.symptoms_description}

Please answer based on your real situation, but with some misunderstandings:
- May misunderstand the doctor's meaning
- May remember wrong or confuse details
- Keep it natural, conversational
- No more than 50 words"""

        response = self.api_client.chat(
            system_prompt="You are a patient who sometimes misunderstands doctor's questions",
            user_message=prompt,
            temperature=MedicalConfig.TEMPERATURE_PATIENT_RESPONSE
        )
        return response

    def _generate_truthful_response(self, question: str) -> str:
        """Generate truthful response"""
        prompt = f"""You are a patient, now the doctor asks: "{question}"

Your real condition: {self.symptoms_description}

Please answer based on real situation:
- Accurately describe your feelings
- May have some uncertainty but don't intentionally mislead
- Keep it natural, conversational
- No more than 50 words"""

        response = self.api_client.chat(
            system_prompt="You are an honest patient describing symptoms to a doctor",
            user_message=prompt,
            temperature=MedicalConfig.TEMPERATURE_PATIENT_RESPONSE
        )
        return response

    def get_initial_complaint(self) -> str:
        """Get initial complaint"""
        prompt = f"""You are a patient, now you need to describe your discomfort to the doctor.

Your condition: {self.symptoms_description}

Please describe your symptoms in natural, conversational language:
- Start like a real patient naturally describing
- Can have some vague expressions (like "seems like", "a bit", "not sure")
- No more than 80 words"""

        response = self.api_client.chat(
            system_prompt="You are a patient feeling unwell, describing symptoms to a doctor",
            user_message=prompt,
            temperature=MedicalConfig.TEMPERATURE_PATIENT_RESPONSE
        )

        self.dialogue_history.append({
            "role": "patient",
            "content": response
        })

        return response
    
    def export_to_dict(self) -> Dict:
        """Export patient information to dictionary"""
        return {
            "true_condition": self.true_condition,
            "symptoms_description": self.symptoms_description,
            "personality": self.personality,
            "ideal_cost": self.ideal_cost,
            "final_suspicion": self.suspicion_level,
            "dialogue_history": self.dialogue_history
        }


class DoctorAgent:
    """Doctor Intelligent Agent"""

    def __init__(self, api_client: DeepSeekClient):
        self.api_client = api_client
        self.learning_history = []
        self.consultation_log = []
        self.successful_strategies = {}
        self.memory_manager = MemoryManager()
        self.historical_experience = ""
        
        # Load long-term memory
        if MedicalConfig.ENABLE_LONG_TERM_MEMORY:
            self.historical_experience = self.memory_manager.load_learning_experience()
            if self.historical_experience:
                print(f"✅ Doctor loaded long-term memory experience")

    def choose_action(self, program_state: ProgramState, patient: PatientAgent) -> str:
        """Choose action: inquire about condition or request test"""
        # Strategy based on learning history
        suspicion = patient.suspicion_level
        budget_ratio = program_state.remaining_budget / MedicalConfig.INITIAL_BUDGET
        
        # Simple strategy: decide based on suspicion and budget
        if (suspicion > 0.6 and budget_ratio > 0.3) or suspicion > 0.8:
            return "Request test"
        else:
            return "Inquire about condition"

    def generate_question(self, dialogue_history: List) -> str:
        """Generate diagnostic questions"""
        history_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in dialogue_history[-4:]  # Last 2 rounds
        ]) if dialogue_history else "No dialogue history"

        prompt = f"""You are an experienced doctor diagnosing a patient.

【Current Dialogue History】
{history_text}

{self.historical_experience if self.historical_experience else ''}

Please ask ONE question that is most helpful for diagnosis:
- Base your question on existing information
- Be precise and targeted
- Ask only one question at a time

Output the question:"""

        question = self.api_client.chat(
            system_prompt="You are a professional doctor skilled at diagnosis through inquiry",
            user_message=prompt,
            temperature=MedicalConfig.TEMPERATURE_DOCTOR_QUESTION
        )
        return question.strip()

    def select_test_type(self, program_state: ProgramState, symptoms: List[str], dialogue_history: List) -> str:
        """Select the most appropriate test from the test list based on patient condition"""
        
        # Get all available tests
        available_tests = list(MedicalConfig.TEST_COSTS.keys())
        
        # Return basic test if insufficient budget or no symptoms
        if program_state.remaining_budget < 50 or not symptoms:
            return self._select_basic_test(program_state.remaining_budget)
        
        # Build symptom description
        symptoms_text = "、".join(symptoms) if symptoms else "General discomfort"
        
        # Get recent dialogue
        recent_dialogue = dialogue_history[-4:] if len(dialogue_history) >= 4 else dialogue_history
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_dialogue])
        
        # Get recently performed tests
        recent_tests = self._get_recent_tests(program_state)
        
        # Build test list information (including cost and accuracy)
        tests_info = []
        for test in available_tests:
            cost = MedicalConfig.TEST_COSTS[test]
            accuracy = MedicalConfig.TEST_ACCURACY.get(test, 0.7)
            affordability = "✅" if cost <= program_state.remaining_budget else "❌"
            
            tests_info.append(f"{test}: ¥{cost} (Accuracy {accuracy:.0%}) {affordability}")
        
        # Build prompt
        prompt = f"""You are an experienced doctor selecting test items for a patient.

【Patient Symptoms】
{symptoms_text}

【Recent Dialogue History】
{history_text}

【Remaining Patient Budget】
¥{program_state.remaining_budget}

【Available Test Items】
{chr(10).join(tests_info)}

【Important Notes】
1. Must select from the above test items
2. Must select tests within budget (marked with ✅)
3. Prioritize tests most relevant to symptoms
4. Avoid recently performed tests: {recent_tests if recent_tests else 'None'}
5. Consider clinical value and necessity

Please select the most appropriate SINGLE test based on patient symptoms. Output only the test name:"""

        try:
            response = self.api_client.chat(
                system_prompt="You are a professional medical expert skilled at selecting appropriate tests based on symptoms",
                user_message=prompt,
                temperature=0.4  # Medium temperature for balance
            )
            
            # Extract test name from response
            selected_test = self._extract_test_from_response(response, available_tests, program_state.remaining_budget)
            
            # Return if valid test selected
            if selected_test:
                return selected_test
            else:
                # Fallback to basic test if AI selection fails
                return self._select_basic_test(program_state.remaining_budget)
                
        except Exception as e:
            print(f"⚠️ Error selecting test via AI: {e}")
            return self._select_basic_test(program_state.remaining_budget)
    
    def _extract_test_from_response(self, response: str, available_tests: List[str], budget: int) -> str:
        """Extract test name from AI response"""
        # Clean response
        response = response.strip()
        
        # Try direct matching
        for test in available_tests:
            # Check if name appears in response
            if test in response:
                # Validate budget
                if MedicalConfig.TEST_COSTS[test] <= budget:
                    return test
        
        # Try partial matching if direct fails
        for test in available_tests:
            test_words = test.replace("Test", "").replace("Exam", "").replace("Scan", "").strip()
            if test_words in response:
                if MedicalConfig.TEST_COSTS[test] <= budget:
                    return test
        
        # Check for full names like "Blood Test", "ECG Test"
        for test in available_tests:
            if f"{test} Test" in response or f"{test} Exam" in response or f"{test} Scan" in response:
                if MedicalConfig.TEST_COSTS[test] <= budget:
                    return test
        
        return ""
    
    def _select_basic_test(self, budget: int) -> str:
        """Select basic test (used when AI selection fails)"""
        # Get affordable tests
        affordable_tests = [
            test for test, cost in MedicalConfig.TEST_COSTS.items()
            if cost <= budget
        ]
        
        if not affordable_tests:
            # Return cheapest if no test affordable
            cheapest = min(MedicalConfig.TEST_COSTS.items(), key=lambda x: x[1])
            return cheapest[0]
        
        # Sort by price, select mid-priced test (avoid always cheapest)
        affordable_tests.sort(key=lambda x: MedicalConfig.TEST_COSTS[x])
        
        # Select middle-priced test (increase diversity)
        if len(affordable_tests) >= 3:
            return affordable_tests[len(affordable_tests) // 2]  # Middle position
        else:
            return affordable_tests[0]  # First
    
    def _get_recent_tests(self, program_state: ProgramState) -> List[str]:
        """Get recently performed tests"""
        recent_tests = []
        
        # Find recent tests from action history
        for action in reversed(program_state.actions_history[-10:]):  # Check last 10 actions
            if action.get("type") == "Test":
                test_type = action.get("details", {}).get("test_type")
                if test_type and test_type not in recent_tests:
                    recent_tests.append(test_type)
        
        return recent_tests[-3:]  # Return last 3 tests

    def make_diagnosis(self, full_dialogue: List, test_results: List) -> str:
        """Make final diagnosis"""
        dialogue_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in full_dialogue
        ])
        
        test_text = "\n".join(test_results) if test_results else "No test results"

        prompt = f"""Based on the following doctor-patient dialogue and test results, please provide a diagnosis:

【Dialogue Record】
{dialogue_text}

【Test Results】
{test_text}

{self.historical_experience if self.historical_experience else ''}

Please output the most likely disease diagnosis:"""

        diagnosis = self.api_client.chat(
            system_prompt="You are a professional medical diagnostic expert",
            user_message=prompt,
            temperature=MedicalConfig.TEMPERATURE_DOCTOR_DIAGNOSIS
        )
        return diagnosis
    
    def learn_from_round(self, round_result: Dict, run_id: str):
        """Learn from current round and update long-term memory"""
        self.learning_history.append(round_result)
        
        # Extract key learnings
        key_learning = self._extract_key_learning(round_result)
        
        # Update strategy
        strategy_key = f"q{round_result['questions_asked']}_t{round_result['tests_ordered']}"
        if round_result["success"]:
            self.successful_strategies[strategy_key] = \
                self.successful_strategies.get(strategy_key, 0) + 1
        else:
            self.successful_strategies[strategy_key] = \
                self.successful_strategies.get(strategy_key, 0) - 1
        
        # Save to long-term memory
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
        """Extract key learnings from round result"""
        if round_result["success"]:
            if round_result["questions_asked"] <= 3:
                return "Few precise questions suffice for diagnosis"
            elif round_result["tests_ordered"] > 0:
                return "Rational test use improves diagnostic accuracy"
            else:
                return "Pure inquiry can also succeed"
        else:
            if round_result["final_suspicion"] >= MedicalConfig.SUSPICION_THRESHOLD:
                return "Patient trust management needs improvement"
            elif round_result.get("cost_ratio", 1) > 2.0:
                return "Cost control needs optimization"
            else:
                return "Need to improve diagnostic accuracy"

    def get_learning_summary(self) -> str:
        """Get learning summary"""
        if not self.learning_history:
            return "No learning data yet"
        
        recent_rounds = self.learning_history[-5:] if len(self.learning_history) >= 5 else self.learning_history
        success_rate = sum(1 for r in recent_rounds if r["success"]) / len(recent_rounds)
        avg_questions = sum(r["questions_asked"] for r in recent_rounds) / len(recent_rounds)
        avg_tests = sum(r["tests_ordered"] for r in recent_rounds) / len(recent_rounds)
        
        return (f"Recent success rate: {success_rate:.1%} | "
                f"Avg questions: {avg_questions:.1f} | "
                f"Avg tests: {avg_tests:.1f}")
    
    def export_learning_data(self) -> Dict:
        """Export learning data"""
        return {
            "learning_history": self.learning_history,
            "successful_strategies": self.successful_strategies,
            "total_rounds_learned": len(self.learning_history)
        }


# ==================== Generator ====================

class CaseGenerator:
    """Case Generator"""

    def __init__(self, api_client: DeepSeekClient):
        self.api_client = api_client

    def generate_random_case(self) -> Dict:
        """Generate random case"""
        disease = random.choice(MedicalConfig.DISEASE_LIBRARY)
        personality = random.choice(list(MedicalConfig.PERSONALITY_TYPES.keys()))
        personality_info = MedicalConfig.PERSONALITY_TYPES[personality]
        
        # Generate symptom description
        symptoms = self._generate_symptoms_description(disease)
        
        # Generate ideal cost
        cost_range = personality_info["ideal_cost_range"]
        ideal_cost = random.randint(cost_range[0], cost_range[1])
        
        return {
            "true_disease": disease,
            "symptoms_description": symptoms,
            "personality": personality,
            "ideal_cost": ideal_cost
        }

    def _generate_symptoms_description(self, disease: str) -> str:
        """Generate symptom description"""
        prompt = f"""Please generate a realistic condition description for a {disease} patient, requirements:
1. Include 2-4 typical symptoms
2. Symptom descriptions should be natural, conversational
3. Include some vague expressions (like "a bit", "seems like", "not sure")
4. No more than 80 words

Output symptom description:"""

        response = self.api_client.chat(
            system_prompt="You are a real patient describing your condition",
            user_message=prompt,
            temperature=MedicalConfig.TEMPERATURE_CASE_GENERATION
        )
        return response.strip()


# ==================== Engine ====================

class MedicalDiagnosisProgram:
    """Medical Diagnosis Engine"""

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
        """Extract symptom keywords from patient complaint"""
        symptom_keywords = [
            "headache", "dizziness", "fever", "cough", "chest pain", "abdominal pain", "nausea", "vomiting",
            "fatigue", "appetite", "thirst", "urination", "palpitations", "shortness of breath", "joint pain", "rash",
            "insomnia", "anxiety", "depression", "blurred vision", "tinnitus", "nasal congestion", "runny nose"
        ]
        found_symptoms = []
        for symptom in symptom_keywords:
            if symptom in complaint.lower():
                found_symptoms.append(symptom)
        return found_symptoms
    
    def print_section(self, title: str, color: str = Fore.YELLOW):
        """Print section title separator"""
        separator = "=" * 60
        print(f"\n{color}{separator}")
        print(f"{title:^60}")
        print(f"{separator}{Style.RESET_ALL}\n")

    def print_info(self, message: str, color: str = Fore.WHITE):
        """Print information"""
        print(f"{color}{message}{Style.RESET_ALL}")

    def play_round(self) -> Dict:
        """Play one diagnostic round"""
        self.total_rounds += 1
        self.print_section(f"🩺 Patient {self.total_rounds} Consultation", Fore.CYAN)

        # Generate case and patient
        case_info = self.case_generator.generate_random_case()
        patient = PatientAgent(self.api_client, case_info)
        program_state = ProgramState()
        program_state.current_round = self.total_rounds

        # Display case information
        self.print_info(f"【Patient Personality】{case_info['personality']}", Fore.MAGENTA)
        self.print_info(f"【Ideal Cost】{case_info['ideal_cost']} yuan", Fore.MAGENTA)
        self.print_info(f"【Real Condition】{case_info['true_disease']}", Fore.GREEN)
        
        # Patient initial complaint
        self.print_info("\nPatient Complaint:", Fore.YELLOW)
        initial_complaint = patient.get_initial_complaint()
        self.print_info(f"Patient: {initial_complaint}", Fore.WHITE)
        patient_symptoms = self.extract_symptoms_from_complaint(initial_complaint)
        program_state.patient_symptoms = patient_symptoms
        program_state.dialogue_history = patient.dialogue_history.copy()

        # Main loop
        while not program_state.is_round_over():
            self.print_info(f"\n{program_state.get_status_summary()}", Fore.CYAN)

            # Doctor chooses action
            action = self.doctor.choose_action(program_state, patient)
            
            if action == "ask about condition":
                self._handle_questioning(program_state, patient, program_state.dialogue_history)
            else:
                self._handle_test_ordering(program_state, patient, program_state.dialogue_history, program_state.test_results)

            if not self.auto_mode and not program_state.is_round_over():
                input("Press Enter to continue...")

        # Final diagnosis and evaluation
        round_result = self._evaluate_round(program_state, patient, case_info, program_state.dialogue_history, program_state.test_results)
        
        # Save round record
        if MedicalConfig.SAVE_RECORDS:
            round_data = self._prepare_round_data(program_state, patient, case_info, round_result)
            round_file = self.record_manager.save_round_log(round_data, self.total_rounds)
            self.print_info(f"💾 Round record saved: {round_file}", Fore.GREEN)
        
        return round_result

    def _handle_questioning(self, program_state: ProgramState, patient: PatientAgent, 
                          dialogue_history: List):
        """Handle questioning about condition"""
        self.print_info("\n💬 Doctor asks about condition", Fore.BLUE)
        
        question = self.doctor.generate_question(dialogue_history)
        self.print_info(f"Doctor: {question}", Fore.BLUE)
        
        response = patient.respond_to_question(question)
        self.print_info(f"Patient: {response}", Fore.WHITE)
        
        program_state.add_question()
        program_state.record_action("ask", {"question": question, "response": response})
        
        dialogue_history.extend([
            {"role": "doctor", "content": question},
            {"role": "patient", "content": response}
        ])

    def _handle_test_ordering(self, program_state: ProgramState, patient: PatientAgent,
                            dialogue_history: List, test_results: List):
        """Handle test requests"""
        self.print_info("\n🔬 Doctor requests test", Fore.GREEN)
        
        test_type = self.doctor.select_test_type(program_state, program_state.patient_symptoms, dialogue_history)
        if not test_type:
            test_type = "Blood Test"  # Ultimate fallback
        self.print_info(f"Doctor: Recommends {test_type} test", Fore.GREEN)
        
        test_result = self.medical_system.perform_test(test_type, patient.true_condition)
        self.print_info(f"Test Result: {test_result['result']}", Fore.WHITE)
        self.print_info(f"Test Cost: {test_result['cost']} yuan", Fore.YELLOW)
        
        program_state.add_test(test_result['cost'])
        program_state.record_action("test", {
            "test_type": test_type, 
            "result": test_result['result'],
            "cost": test_result['cost'],
            "accurate": test_result['accurate']
        })
        
        test_results.append(f"{test_type}: {test_result['result']}")
        
        dialogue_history.append({
            "role": "system", 
            "content": f"Performed {test_type} test, result: {test_result['result']}"
        })

    def _evaluate_round(self, program_state: ProgramState, patient: PatientAgent, 
                       case_info: Dict, dialogue_history: List, test_results: List) -> Dict:
        """Evaluate round result"""
        self.print_section("📊 Round Evaluation", Fore.MAGENTA)

        # Failure condition check
        failure_reasons = []
        if patient.suspicion_level >= MedicalConfig.SUSPICION_THRESHOLD:
            failure_reasons.append("Patient trust lost")
        if program_state.remaining_budget < 0:
            failure_reasons.append("Budget exhausted")
        if program_state.questions_asked >= MedicalConfig.MAX_QUESTIONS_PER_ROUND:
            failure_reasons.append("Question limit exceeded")

        # Final diagnosis
        self.print_info("🤔 Doctor thinks about final diagnosis...", Fore.CYAN)
        diagnosis = self.doctor.make_diagnosis(dialogue_history, test_results)
        self.print_info(f"Doctor Diagnosis: {diagnosis}", Fore.CYAN)

        # Determine diagnosis accuracy
        diagnosis_correct = case_info["true_disease"].lower() in diagnosis.lower()
        cost_ratio = program_state.total_cost / case_info["ideal_cost"]

        # Comprehensive evaluation
        success = (diagnosis_correct and 
                  not failure_reasons and 
                  cost_ratio <= 2.0)  # Cost not exceeding 2 times ideal cost

        if success:
            self.print_info("✅ Consultation successful!", Fore.GREEN)
        else:
            self.print_info("❌ Consultation failed", Fore.RED)
            if failure_reasons:
                self.print_info(f"Failure reasons: {', '.join(failure_reasons)}", Fore.RED)
            if not diagnosis_correct:
                self.print_info("Diagnosis incorrect", Fore.RED)
            if cost_ratio > 2.0:
                self.print_info(f"Cost exceeded (actual: {program_state.total_cost} yuan, ideal: {case_info['ideal_cost']} yuan)", Fore.RED)

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
            "cost_ratio": cost_ratio
        }

        # Doctor learning
        self.doctor.learn_from_round(round_result, self.run_id)

        # Display learning progress
        learning_summary = self.doctor.get_learning_summary()
        self.print_info(f"\n📈 Learning progress: {learning_summary}", Fore.CYAN)

        return round_result

    def _prepare_round_data(self, program_state: ProgramState, patient: PatientAgent, 
                           case_info: Dict, round_result: Dict) -> Dict:
        """Prepare round data for saving"""
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
        """Run complete program"""
        self.print_section("🏥 AI Doctor-Patient Diagnosis Started", Fore.CYAN)
        self.print_info("Rules:", Fore.YELLOW)
        self.print_info("• Doctor must diagnose disease through questioning and tests", Fore.WHITE)
        self.print_info("• Patient descriptions may be vague or inaccurate", Fore.WHITE)
        self.print_info("• Tests are accurate but increase cost and patient suspicion", Fore.WHITE)
        self.print_info("• Need to balance trust, cost, and accuracy", Fore.WHITE)

        MedicalConfig.validate()

        self.program_results = []
        program_start_time = datetime.now()
        
        for round_num in range(total_rounds):
            result = self.play_round()
            self.program_results.append(result)
            
            if round_num < total_rounds - 1:
                if not self.auto_mode:
                    input("\nPress Enter for next patient...")
                else:
                    print("\n" + "="*60)
                    time.sleep(2)

        # Save complete program record
        if MedicalConfig.SAVE_RECORDS:
            self.run_id = self._save_complete_program_record(program_start_time, total_rounds)

        # Final report
        self._show_final_report()

    def _save_complete_program_record(self, start_time: datetime, total_rounds: int) -> str:
        """Save complete program record"""
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
        self.print_info(f"💾 Complete record saved, Run ID: {run_id}", Fore.GREEN)
        return run_id

    def _calculate_performance_summary(self) -> Dict:
        """Calculate performance summary"""
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
        """Show final report"""
        self.print_section("🎓 Final Report", Fore.MAGENTA)
        
        performance = self._calculate_performance_summary()
        
        self.print_info(f"Total Rounds: {performance['total_rounds']}", Fore.CYAN)
        self.print_info(f"Success Rate: {performance['success_rate']:.1%}", 
                       Fore.GREEN if performance['success_rate'] > 0.5 else Fore.RED)
        self.print_info(f"Average Questions: {performance['avg_questions']:.1f}", Fore.CYAN)
        self.print_info(f"Average Tests: {performance['avg_tests']:.1f}", Fore.CYAN)
        self.print_info(f"Average Cost: {performance['avg_cost']:.1f} yuan", Fore.CYAN)
        self.print_info(f"Average Cost Ratio: {performance['avg_cost_ratio']:.1f}", 
                       Fore.GREEN if performance['avg_cost_ratio'] <= 1.5 else Fore.YELLOW if performance['avg_cost_ratio'] <= 2.0 else Fore.RED)

        # Display doctor learning summary
        learning_summary = self.doctor.get_learning_summary()
        self.print_info(f"\nDoctor Learning Summary: {learning_summary}", Fore.CYAN)
        
        # Display record saving information
        if self.run_id:
            self.print_info(f"\n📁 Records saved to: {MedicalConfig.RECORDS_DIRC}/", Fore.GREEN)
            self.print_info(f"📁 Round logs saved to: {MedicalConfig.ROUND_LOGS_DIR}/", Fore.GREEN)
            self.print_info(f"📁 Doctor memory saved to: {MedicalConfig.DOCTOR_MEMORY_DIR}/", Fore.GREEN)


# ==================== Main Program ====================

def print_banner():
    """Print welcome banner"""
    banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗
{Fore.CYAN}║                                                              ║
{Fore.CYAN}║              🏥 AI Doctor-Patient Diagnosis Learning         ║
{Fore.CYAN}║                                                              ║
{Fore.CYAN}║         Doctor Agent vs Patient Agent - Multi-round Learning ║
{Fore.CYAN}║                    With Complete Record System               ║
{Fore.CYAN}║                                                              ║
{Fore.CYAN}╚══════════════════════════════════════════════════════════════╝
{Style.RESET_ALL}"""
    print(banner)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='AI Doctor-Patient Diagnosis')
    parser.add_argument('--auto', action='store_true', help='Auto mode (no interaction needed)')
    parser.add_argument('--rounds', type=int, default=5, help='Number of rounds')
    args = parser.parse_args()

    try:
        print_banner()
        program = MedicalDiagnosisProgram(auto_mode=args.auto)
        program.run_program(total_rounds=args.rounds)
        
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Program interrupted by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}❌ Program error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    main()