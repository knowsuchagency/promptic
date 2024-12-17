from datetime import datetime

from promptic import llm


@llm(model="gpt-4o")
def scheduler(command):
    """{command}"""


@scheduler.tool
def get_current_time():
    """Get the current time"""
    print("getting current time")
    return datetime.now().strftime("%I:%M %p")


@scheduler.tool
def add_reminder(task: str, time: str):
    """Add a reminder for a specific task and time"""
    print(f"adding reminder: {task} at {time}")
    return f"Reminder set: {task} at {time}"


@scheduler.tool
def check_calendar(date: str):
    """Check calendar for a specific date"""
    print(f"checking calendar for {date}")
    return f"Calendar checked for {date}: No conflicts found"


cmd = """
What time is it?
Also, can you check my calendar for tomorrow
and set a reminder for a team meeting at 2pm?
"""

print(scheduler(cmd))
# getting current time
# checking calendar for 2023-10-05
# adding reminder: Team meeting at 2023-10-05T14:00:00
# The current time is 3:48 PM. I checked your calendar for tomorrow, and there are no conflicts. I've also set a reminder for your team meeting at 2 PM tomorrow.
