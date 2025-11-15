import { create } from "zustand";

type StoreTypes = {
  scenarioState: keyof typeof scenarioStates;
  unlockedCamera: boolean;
  unlockedMotion: boolean;
  unlockedFire: boolean;
  unlockedTracking: boolean;
  unlockedAutoFire: boolean;
  unlockedMotionPrediction: boolean;
  unlockedCode: boolean;
  unlockedCredits: boolean;
  tab: "video" | "code" | "credits";
  activeCamera: boolean;
  activeMotionDetection: boolean;
  activeFire: boolean;
  activeTracking: boolean;
  activeAutoFire: boolean;
  activeMotionPrediction: boolean;
  typingSpeedIndex: number; // 0-5, where 0 is slowest and 5 is fastest
  codeProgress: number; // 0-100, progress of code typing
  codeTypedCode: string; // The actual typed code string
  ttsQueue: string[]; // Queue of messages from commander to be spoken via text-to-speech
  setScenarioState: (value: keyof typeof scenarioStates) => void;
  setUnlockedCamera: (value: boolean) => void;
  setUnlockedMotion: (value: boolean) => void;
  setUnlockedFire: (value: boolean) => void;
  setUnlockedTracking: (value: boolean) => void;
  setUnlockedAutoFire: (value: boolean) => void;
  setUnlockedMotionPrediction: (value: boolean) => void;
  setUnlockedCode: (value: boolean) => void;
  setUnlockedCredits: (value: boolean) => void;
  setTab: (value: "video" | "code" | "credits") => void;
  setActiveCamera: (value: boolean) => void;
  setActiveMotionDetection: (value: boolean) => void;
  setActiveFire: (value: boolean) => void;
  setActiveTracking: (value: boolean) => void;
  setActiveAutoFire: (value: boolean) => void;
  setActiveMotionPrediction: (value: boolean) => void;
  setTypingSpeedIndex: (value: number) => void;
  setCodeProgress: (value: number) => void;
  setCodeTypedCode: (value: string) => void;
  addToTtsQueue: (message: string) => void;
  popFromTtsQueue: () => string | undefined;
  sendMessageCallback: ((action: string) => void) | null;
  setSendMessageCallback: (callback: ((action: string) => void) | null) => void;
  triggerAction: (action: string) => void;
};

export const scenarioStates = {
  intro:
    "You are the Commander guiding the user in a high-security military drone defense simulation. The mission is classified. The first step is to introduce yourself and you need to unlock the camera feed for the user. Instruct the user to toggle ON the camera feed to start the mission.",
  mission:
    "With the camera feed active, provide the user with crucial mission context. Inform them that their task is to eliminate an enemy drone breaching our airspace. Emphasize the urgency and threat the drone poses. After briefing the mission, immediately unlock the 'Fire' ability using the unlockFire tool so the user can attempt to shoot the drone.",
  fireUnlocked:
    "The 'Fire' ability has been unlocked. Instruct the user to toggle on 'left click to fire' and attempt to shoot the enemy drone manually. Note that the drone moves too quickly for manual targeting, and prompt the user to experience this difficulty. After they try and fail, unlock the Code tab using the unlockCode tool.",
  codeUnlocked:
    "The Code tab has been unlocked. Clearly instruct the user to open the Code tab to develop an automated interception solution.",
  firstCode:
    "The user has entered the Code tab. As the Commander, task them with writing software to extract and track the drone's position from the camera feed (motion detection). Clearly specify that completion of this code is required before proceeding.",
  motionDetection:
    "The motion detection code has been completed and the 'Motion Detection' toggle has been automatically unlocked. Instruct the user to enable motion detection by toggling it on.",
  tracking:
    "The next step is critical. Instruct the user to create a tracking program that leverages the motion detection data to accurately follow the drone's position. Wait for the user to complete this tracking code.",
  trackingComplete:
    "The tracking code has been completed and the 'Tracking' toggle has been automatically unlocked. Guide the user to enable tracking, which allows continuous crosshair lock on the drone. Suggest that the user test the tracking system and attempt to fire.",
  positionPrediction:
    "If shots are missing due to drone speed, explain to the user that they must implement position prediction using tracking data. Instruct the user to write code that predicts where the drone will be, and wait for them to finish this stage.",
  positionPredictionComplete:
    "The position prediction code has been completed and the 'Position Prediction' toggle has been automatically unlocked. Instruct the user to activate this feature, improving crosshair accuracy. Encourage another attempt at firing.",
  success:
    "The user has successfully intercepted the drone. Congratulate them formally and use the unlockCredits tool to unlock the Credits tab as a reward.",
};

export const useStore = create<StoreTypes>((set) => ({
  scenarioState: "intro",
  unlockedCamera: false,
  unlockedMotion: false,
  unlockedFire: false,
  unlockedTracking: false,
  unlockedAutoFire: false,
  unlockedMotionPrediction: false,
  unlockedCode: false,
  unlockedCredits: false,
  tab: "video",
  activeCamera: false,
  activeMotionDetection: false,
  activeFire: false,
  activeTracking: false,
  activeAutoFire: false,
  activeMotionPrediction: false,
  typingSpeedIndex: 0, // Start at slowest speed
  codeProgress: 0, // Start at 0% progress
  codeTypedCode: "", // Start with empty code
  ttsQueue: [], // Start with empty TTS queue
  setScenarioState: (value) => set({ scenarioState: value }),
  setUnlockedCamera: (value) => set({ unlockedCamera: value }),
  setUnlockedMotion: (value) => set({ unlockedMotion: value }),
  setUnlockedFire: (value) => {
    set((state) => {
      // Progress state: mission → fireUnlocked
      if (value && state.scenarioState === "mission") {
        return {
          unlockedFire: value,
          scenarioState: "fireUnlocked" as const,
        };
      }
      return { unlockedFire: value };
    });
  },
  setUnlockedTracking: (value) => set({ unlockedTracking: value }),
  setUnlockedAutoFire: (value) => set({ unlockedAutoFire: value }),
  setUnlockedMotionPrediction: (value) =>
    set({ unlockedMotionPrediction: value }),
  setUnlockedCode: (value) => {
    set((state) => {
      // Progress state: fireUnlocked → codeUnlocked
      if (value && state.scenarioState === "fireUnlocked") {
        return {
          unlockedCode: value,
          scenarioState: "codeUnlocked" as const,
        };
      }
      return { unlockedCode: value };
    });
  },
  setUnlockedCredits: (value) => {
    set((state) => {
      // Progress state: positionPredictionComplete → success
      if (value && state.scenarioState === "positionPredictionComplete") {
        return {
          unlockedCredits: value,
          scenarioState: "success" as const,
        };
      }
      return { unlockedCredits: value };
    });
  },
  setTab: (value) => {
    set((state) => {
      // codeUnlocked → firstCode: When Code tab is opened
      if (
        value === "code" &&
        state.scenarioState === "codeUnlocked" &&
        state.unlockedCode
      ) {
        setTimeout(() => {
          useStore.getState().triggerAction("opened the code tab");
        }, 0);
        return {
          tab: value,
          scenarioState: "firstCode",
        };
      }
      return { tab: value };
    });
  },
  setActiveCamera: (value) => {
    set((state) => {
      // Only allow camera activation if it's unlocked
      if (value && !state.unlockedCamera) {
        return state; // Don't change state if camera is locked
      }
      // If camera is being turned on and we're in intro state, progress to mission
      if (value && state.scenarioState === "intro") {
        const newState = {
          activeCamera: value,
          scenarioState: "mission" as const,
        };
        // Trigger action after state update
        setTimeout(() => {
          useStore.getState().triggerAction("activated the camera");
        }, 0);
        return newState;
      }
      return { activeCamera: value };
    });
  },
  setActiveMotionDetection: (value) => {
    set((state) => {
      // motionDetection → tracking: When motion detection is toggled on
      if (value && state.scenarioState === "motionDetection") {
        setTimeout(() => {
          useStore.getState().triggerAction("enabled motion detection");
        }, 0);
        return {
          activeMotionDetection: value,
          activeCamera: true,
          scenarioState: "tracking",
        };
      }
      if (value) {
        return {
          activeMotionDetection: value,
          activeCamera: true,
        };
      }
      return {
        activeMotionDetection: value,
      };
    });
  },
  setActiveFire: (value) => set({ activeFire: value }),
  setActiveTracking: (value) => {
    set((state) => {
      // trackingComplete → positionPrediction: When tracking is enabled
      if (value && state.scenarioState === "trackingComplete") {
        setTimeout(() => {
          useStore.getState().triggerAction("enabled tracking");
        }, 0);
        return {
          activeTracking: value,
          scenarioState: "positionPrediction",
        };
      }
      return { activeTracking: value };
    });
  },
  setActiveAutoFire: (value) => set({ activeAutoFire: value }),
  setActiveMotionPrediction: (value) => set({ activeMotionPrediction: value }),
  setTypingSpeedIndex: (value) => set({ typingSpeedIndex: value }),
  setCodeProgress: (value) => set({ codeProgress: value }),
  setCodeTypedCode: (value) => set({ codeTypedCode: value }),
  addToTtsQueue: (message) =>
    set((state) => ({
      ttsQueue: [...state.ttsQueue, message],
    })),
  popFromTtsQueue: () => {
    let message: string | undefined;
    set((state) => {
      if (state.ttsQueue.length === 0) {
        message = undefined;
        return state;
      }
      const [first, ...rest] = state.ttsQueue;
      message = first;
      return { ttsQueue: rest };
    });
    return message;
  },
  sendMessageCallback: null,
  setSendMessageCallback: (callback) => set({ sendMessageCallback: callback }),
  triggerAction: (action) => {
    // Get current state to access the callback
    const currentState = useStore.getState();
    if (currentState.sendMessageCallback) {
      currentState.sendMessageCallback(action);
    }
  },
}));

// Export getState for use in callbacks
export const getStoreState = () => useStore.getState();
