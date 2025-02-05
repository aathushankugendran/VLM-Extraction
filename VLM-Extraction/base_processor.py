# base_processor.py

# Import ABC (Abstract Base Class) and abstractmethod to enforce OOP principles
from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    """
    BaseProcessor is an abstract class that defines a standard interface 
    for all processing components in the pipeline.
    
    Any subclass must implement the 'process' method to en