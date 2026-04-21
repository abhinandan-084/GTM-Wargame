import re
import numpy as np
from typing import Dict, Any, List

class ConsistencyChecker:
    """
    Principal Engineer Note: This version handles NumPy scalars, 
    NumPy arrays, and Pandas types that often fail standard 
    Python type checks.
    """

    @staticmethod
    def _flatten_data(data: Any) -> List[float]:
        """
        Recursively extracts every number, handling Python natives, 
        NumPy scalars, and NumPy arrays.
        """
        nums = []
        
        # 1. Handle NumPy arrays or Python lists
        if isinstance(data, (list, tuple, np.ndarray)):
            for item in data:
                nums.extend(ConsistencyChecker._flatten_data(item))
        
        # 2. Handle Dictionaries
        elif isinstance(data, dict):
            for v in data.values():
                nums.extend(ConsistencyChecker._flatten_data(v))
        
        # 3. Handle Numbers (Python int/float AND NumPy int/float)
        elif isinstance(data, (int, float, np.number)):
            # Convert to standard float, filter out NaNs or Infs
            val = float(data)
            if np.isfinite(val):
                nums.append(val)
        
        # 4. Handle Booleans (Often used in market_context)
        elif isinstance(data, bool):
            nums.append(float(data))

        return nums

    @staticmethod
    def validate_response(text: str, shap_info: Dict, market_context: Dict, opt_results: Dict) -> Dict[str, Any]:
        """
        Validates the Agent response against a unified Ground Truth pool.
        """
        # 1. Build the Ground Truth Pool from all sources
        raw_pool = []
        for source in [shap_info, market_context, opt_results]:
            raw_pool.extend(ConsistencyChecker._flatten_data(source))
        
        # Use a set for faster lookup and unique values
        ground_truth_pool = [abs(v) for v in raw_pool]

        # 2. Clean and extract numbers from LLM text
        # Remove commas and currency symbols first
        clean_text = text.replace(',', '').replace('$', '').replace('%', '')
        found_in_text = re.findall(r"[-+]?\d*\.\d+|\d+", clean_text)
        
        # Convert to float and absolute value
        text_nums = []
        for n in found_in_text:
            try:
                text_nums.append(abs(float(n)))
            except ValueError:
                continue

        hallucinations = []
        # Tolerance: 1.1 allows for rounding (e.g., 799.8 in data vs 800 in text)
        tolerance = 1.1 

        for num in text_nums:
            # Skip indices and common low-value markdown formatting numbers
            if num in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
                continue
            
            match_found = False
            for gt in ground_truth_pool:
                # Direct/Rounded match
                if abs(num - gt) < tolerance:
                    match_found = True; break
                
                # Percentage scaling (LLM says 5 for data 0.05)
                if abs(num - (gt * 100)) < tolerance:
                    match_found = True; break
                
                # Inverse Scaling (LLM says 0.12 for data 12.0)
                if abs(num - (gt / 100)) < tolerance:
                    match_found = True; break
            
            if not match_found:
                hallucinations.append(num)

        return {
            "is_valid": len(hallucinations) == 0,
            "hallucinated_values": hallucinations,
            "error_msg": f"Detected numbers not in Ground Truth: {hallucinations}" if hallucinations else None
        }