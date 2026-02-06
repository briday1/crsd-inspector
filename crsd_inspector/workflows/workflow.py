"""
Base Workflow class for CRSD Inspector workflows

Example usage:
    workflow = Workflow("My Workflow", "Does interesting things")
    
    # Results can be added in any order:
    workflow.add_text("Starting analysis...")
    workflow.add_table("Results", {"Column": [1, 2, 3]})
    workflow.add_plot(fig1)
    workflow.add_plot(fig2)
    workflow.add_text("Analysis complete")
    workflow.add_table("Summary", {"Metric": ["A", "B"]})
    
    # Build converts to legacy format (tables/plots/text grouped)
    return workflow.build()
"""


class Workflow:
    """
    Base class for CRSD Inspector workflows.
    
    Allows flexible ordering of results (tables, plots, text) by appending them
    in the desired sequence.
    """
    
    def __init__(self, name, description):
        """
        Initialize a workflow.
        
        Args:
            name: The workflow display name
            description: Brief description of what the workflow does
        """
        self.name = name
        self.description = description
        self._results = []
    
    def add_text(self, content):
        """
        Add text content to the workflow results.
        
        Args:
            content: String or list of strings to display
        """
        if isinstance(content, str):
            content = [content]
        self._results.append({"type": "text", "content": content})
        return self
    
    def add_table(self, title, data):
        """
        Add a table to the workflow results.
        
        Args:
            title: Table title
            data: Dict with column names as keys and lists of values
        """
        self._results.append({"type": "table", "title": title, "data": data})
        return self
    
    def add_plot(self, figure):
        """
        Add a plot to the workflow results.
        
        Args:
            figure: Plotly figure object
        """
        self._results.append({"type": "plot", "figure": figure})
        return self
    
    def build(self):
        """
        Build the final results in order.
        
        Returns:
            Dict with "results" key containing ordered list of items,
            each with "type" and type-specific keys (title/data for tables,
            figure for plots, content for text)
        """
        return {"results": self._results}
    
    def clear(self):
        """Clear all results"""
        self._results = []
        return self
