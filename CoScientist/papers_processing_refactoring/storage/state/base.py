from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class BaseStateManager(ABC):
    @abstractmethod
    def get_status(self, article_id: str, step: str) -> Optional[str]:
        pass

    @abstractmethod
    def set_status(self, article_id: str, step: str, status: str, error: Optional[str] = None):
        pass

    @abstractmethod
    def list_states(self, article_id: Optional[str] = None, status: Optional[str] = None, step: Optional[str] = None) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def clear_data(self, article_id: Optional[str] = None):
        pass

    @abstractmethod
    def reset_running_states(self, message: str = "Interrupted"):
        pass

    @abstractmethod
    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()