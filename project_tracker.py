import pandas as pd
import json
import os
from datetime import datetime, timedelta

class ProjectTracker:
    """A class to generate project tracking information for integration with project management tools"""
    
    def __init__(self, project_name="Smithsonian Linked Data Explorer"):
        self.project_name = project_name
        self.tasks = []
        self.start_date = datetime.now()
        
    def add_task(self, task_name, description, status="Not Started", 
                 priority="Medium", assigned_to="Me", 
                 duration_days=1, dependencies=None):
        """Add a task to the project tracker"""
        
        # Calculate dates
        if not self.tasks:
            start_date = self.start_date
        else:
            # If there are dependencies, start after the last dependency
            if dependencies:
                dep_end_dates = [self.tasks[dep]["end_date"] for dep in dependencies if dep < len(self.tasks)]
                if dep_end_dates:
                    start_date = max(dep_end_dates)
                else:
                    start_date = self.start_date
            else:
                # Otherwise, start after the previous task
                start_date = self.tasks[-1]["end_date"] if self.tasks else self.start_date
                
        end_date = start_date + timedelta(days=duration_days)
        
        task = {
            "id": len(self.tasks) + 1,
            "name": task_name,
            "description": description,
            "status": status,
            "priority": priority,
            "assigned_to": assigned_to,
            "start_date": start_date,
            "end_date": end_date,
            "dependencies": dependencies or []
        }
        
        self.tasks.append(task)
        return task["id"]
    
    def to_dataframe(self):
        """Convert tasks to a pandas DataFrame"""
        return pd.DataFrame(self.tasks)
    
    def export_to_csv(self, filename="project_tasks.csv"):
        """Export tasks to a CSV file"""
        df = self.to_dataframe()
        
        # Convert datetime objects to strings for CSV export
        df["start_date"] = df["start_date"].apply(lambda x: x.strftime("%Y-%m-%d"))
        df["end_date"] = df["end_date"].apply(lambda x: x.strftime("%Y-%m-%d"))
        
        os.makedirs('project_management', exist_ok=True)
        filepath = os.path.join('project_management', filename)
        df.to_csv(filepath, index=False)
        return filepath
    
    def export_to_json(self, filename="project_tasks.json"):
        """Export tasks to a JSON file that can be imported into project management tools"""
        tasks_export = []
        
        for task in self.tasks:
            task_export = task.copy()
            # Convert datetime objects to strings
            task_export["start_date"] = task["start_date"].strftime("%Y-%m-%d")
            task_export["end_date"] = task["end_date"].strftime("%Y-%m-%d")
            tasks_export.append(task_export)
        
        project_data = {
            "project": {
                "name": self.project_name,
                "start_date": self.start_date.strftime("%Y-%m-%d"),
                "tasks": tasks_export
            }
        }
        
        os.makedirs('project_management', exist_ok=True)
        filepath = os.path.join('project_management', filename)
        with open(filepath, 'w') as f:
            json.dump(project_data, f, indent=2)
        return filepath
    
    def create_airtable_import_file(self, filename="airtable_import.csv"):
        """Create a CSV file formatted for import into Airtable"""
        df = self.to_dataframe()
        
        # Format for Airtable
        airtable_df = pd.DataFrame({
            "Task Name": df["name"],
            "Description": df["description"],
            "Status": df["status"],
            "Priority": df["priority"],
            "Assigned To": df["assigned_to"],
            "Start Date": df["start_date"].apply(lambda x: x.strftime("%Y-%m-%d")),
            "End Date": df["end_date"].apply(lambda x: x.strftime("%Y-%m-%d")),
            "Dependencies": df["dependencies"].apply(lambda x: ", ".join([str(dep) for dep in x]) if x else "")
        })
        
        os.makedirs('project_management', exist_ok=True)
        filepath = os.path.join('project_management', filename)
        airtable_df.to_csv(filepath, index=False)
        return filepath