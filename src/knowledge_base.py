import chromadb

# Initialize vector database
client = chromadb.Client()
collection = client.create_collection(name="fitness_exercises")

# Sample exercise data
exercises = [
    {"id": "1", "name": "Push-ups", "difficulty": "beginner", "muscles": "chest, shoulders, triceps", "instructions": "Place hands shoulder-width apart, lower body until chest nearly touches floor, push back up"},
    {"id": "2", "name": "Squats", "difficulty": "beginner", "muscles": "legs, glutes, quads", "instructions": "Feet shoulder-width apart, lower hips back and down, keep chest up, return to standing"},
    {"id": "3", "name": "Deadlifts", "difficulty": "intermediate", "muscles": "back, glutes, hamstrings", "instructions": "Feet hip-width apart, grip bar, drive through heels to stand up, lower with control"},
    {"id": "4", "name": "Bench Press", "difficulty": "intermediate", "muscles": "chest, shoulders, triceps", "instructions": "Lie flat, grip bar shoulder-width, lower to chest, press back up"},
    {"id": "5", "name": "Pull-ups", "difficulty": "intermediate", "muscles": "back, biceps, shoulders", "instructions": "Grip bar with hands shoulder-width apart, pull body up until chin above bar, lower with control"},
    {"id": "6", "name": "Planks", "difficulty": "beginner", "muscles": "core, shoulders", "instructions": "Forearms on ground, body in straight line, hold position without sagging"},
    {"id": "7", "name": "Running", "difficulty": "beginner", "muscles": "legs, cardio", "instructions": "Maintain steady pace, keep posture upright, breathe regularly"},
    {"id": "8", "name": "Burpees", "difficulty": "advanced", "muscles": "full body, cardio", "instructions": "Squat, place hands down, jump feet back, do push-up, jump feet forward, jump up"},
]

# Add exercises to collection
for exercise in exercises:
    collection.add(
        ids=[exercise["id"]], 
        documents=[f"{exercise['name']}: {exercise['instructions']}"],
        metadatas=[{"difficulty": exercise["difficulty"], "muscles": exercise["muscles"]}]
    )

def retrieve_relevant_exercises(query, n_results=3):
    """Search for relevant exercises based on user query"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    if results['documents']:
        return "\n".join(results['documents'][0])
    return "No relevant exercises found"
