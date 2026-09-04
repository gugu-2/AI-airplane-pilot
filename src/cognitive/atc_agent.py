import re

class ATCNaturalLanguageProcessor:
    """
    Cognitive Layer: NLP Agent for Air Traffic Control.
    In a full production environment, this would use a local LLM (like Llama-3-8B) 
    or a Whisper speech-to-text model. 
    Here we implement a robust keyword/intent extraction engine specifically 
    tuned for standard aviation phraseology.
    """
    def __init__(self, callsign="Aegis 1"):
        self.callsign = callsign.lower()
        print(f"[Cognitive-ATC] NLP Agent initialized. Listening for callsign '{self.callsign}'...")

        # Define mission-critical intents and their regex patterns
        self.intents = {
            "TAKEOFF_CLEARANCE": r"(cleared for takeoff|cleared for departure)",
            "LANDING_CLEARANCE": r"(cleared to land|cleared for the visual)",
            "HOLD_POSITION": r"(hold short|hold position|maintain present heading)",
            "ABORT_LANDING": r"(go around|abort landing|pull up)",
            "ALTITUDE_CHANGE": r"(climb and maintain|descend and maintain)\s*(\d+)",
        }

    def process_audio_transcript(self, transcript: str):
        """
        Parses an incoming ATC transcript and extracts the operational intent.
        """
        transcript = transcript.lower()
        
        # 1. Check if the message is directed at us
        if self.callsign not in transcript:
            return {"intent": "IGNORE", "reason": "Not addressed to our callsign."}

        # 2. Extract Intent
        for intent_name, pattern in self.intents.items():
            match = re.search(pattern, transcript)
            if match:
                result = {"intent": intent_name, "raw_text": transcript}
                
                # Extract specific parameters if applicable (e.g., Altitude)
                if intent_name == "ALTITUDE_CHANGE":
                    target_alt = int(match.group(2))
                    result["target_altitude"] = target_alt
                    
                return result

        # 3. Fallback for unrecognized commands
        return {"intent": "UNKNOWN", "raw_text": transcript, "reason": "Could not parse aviation intent."}

    def generate_readback(self, parsed_intent: dict):
        """
        Aviation rules require the pilot (or AI) to read back the clearance 
        to ATC to confirm understanding.
        """
        intent = parsed_intent.get("intent")
        
        if intent == "TAKEOFF_CLEARANCE":
            return f"{self.callsign.title()}, cleared for takeoff."
        elif intent == "LANDING_CLEARANCE":
            return f"{self.callsign.title()}, cleared to land."
        elif intent == "HOLD_POSITION":
            return f"{self.callsign.title()}, holding position."
        elif intent == "ABORT_LANDING":
            return f"{self.callsign.title()}, going around."
        elif intent == "ALTITUDE_CHANGE":
            alt = parsed_intent.get("target_altitude", "assigned")
            return f"{self.callsign.title()}, climbing to {alt}."
        else:
            return f"{self.callsign.title()}, say again?"

if __name__ == "__main__":
    print("Testing NLP Air Traffic Control Engine:\n")
    atc_nlp = ATCNaturalLanguageProcessor(callsign="Aegis 1")
    
    test_transcripts = [
        "Delta 452, cleared for takeoff runway 24.", # Should ignore
        "Aegis 1, cleared for takeoff.",            # Should parse Takeoff
        "Tower to Aegis 1, traffic on runway, go around! I repeat, go around!", # Should parse Abort
        "Aegis 1, climb and maintain 5000.",        # Should parse Altitude and extract 5000
        "Aegis 1, turn left heading 270."           # Should parse Unknown (not implemented in MVP)
    ]
    
    for text in test_transcripts:
        print(f"\n[ATC Audio In] : \"{text}\"")
        parsed = atc_nlp.process_audio_transcript(text)
        
        if parsed["intent"] == "IGNORE":
            print(f"[AI Brain]     : Ignoring (Addressed to another aircraft).")
        else:
            print(f"[AI Brain]     : Extracted Intent -> {parsed['intent']}")
            if "target_altitude" in parsed:
                print(f"[AI Brain]     : Target Altitude -> {parsed['target_altitude']} ft")
            
            readback = atc_nlp.generate_readback(parsed)
            print(f"[Radio Out]    : \"{readback}\"")
