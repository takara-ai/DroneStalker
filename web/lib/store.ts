import { create } from "zustand";
import { highlightId } from "./highlight";

type CodeKey = "motionDetection" | "tracking" | "positionPrediction";

type CodeState = {
  unlocked: boolean;
  typingSpeedIndex: number; // 0-3, where 0 is slowest and 3 is fastest
  codeProgress: number; // 0-100, progress of code typing
  codeTypedCode: string; // The actual typed code string
};

type StoreTypes = {
  scenarioState: keyof typeof scenarioStates;
  dataId: string; // ID of the data set being used (e.g., "230")
  unlockedCamera: boolean;
  unlockedMotion: boolean;
  unlockedFire: boolean;
  unlockedTracking: boolean;
  unlockedAutoFire: boolean;
  unlockedMotionPrediction: boolean;
  unlockedCode: boolean;
  unlockedCredits: boolean;
  tab:
    | "video"
    | "motionDetection"
    | "tracking"
    | "positionPrediction"
    | "credits";
  activeCamera: boolean;
  activeMotionDetection: boolean;
  activeFire: boolean;
  activeTracking: boolean;
  activeAutoFire: boolean;
  activeMotionPrediction: boolean;
  activeLockTarget: boolean;
  droneHealth: number; // 0-100, drone health percentage
  codeStates: Record<CodeKey, CodeState>;
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
  setDataId: (value: string) => void;
  setTab: (
    value:
      | "video"
      | "motionDetection"
      | "tracking"
      | "positionPrediction"
      | "credits"
  ) => void;
  setActiveCamera: (value: boolean) => void;
  setActiveMotionDetection: (value: boolean) => void;
  setActiveFire: (value: boolean) => void;
  setActiveTracking: (value: boolean) => void;
  setActiveAutoFire: (value: boolean) => void;
  setActiveMotionPrediction: (value: boolean) => void;
  setActiveLockTarget: (value: boolean) => void;
  setDroneHealth: (value: number) => void;
  setCodeUnlocked: (codeKey: CodeKey, value: boolean) => void;
  setCodeTypingSpeedIndex: (codeKey: CodeKey, value: number) => void;
  setCodeProgress: (codeKey: CodeKey, value: number) => void;
  setCodeTypedCode: (codeKey: CodeKey, value: string) => void;
  addToTtsQueue: (message: string) => void;
  popFromTtsQueue: () => string | undefined;
  sendMessageCallback: ((action: string) => void) | null;
  setSendMessageCallback: (callback: ((action: string) => void) | null) => void;
  triggerAction: (action: string) => void;
  unlockAll: () => void;
  reset: () => void;
};

export const scenarioStates = {
  intro:
    "You are the Commander guiding the user in a high-security military drone defense simulation. The mission is classified. The first step is to introduce yourself and instruct the user to toggle ON the camera feed to start the mission.",
  mission:
    "With the camera feed active, provide the user with crucial mission context. Inform them that their task is to eliminate an enemy drone breaching our airspace. Emphasize the urgency and threat the drone poses. The 'Fire' ability is already unlocked. Instruct the user to toggle on 'left click to fire' and attempt to shoot the enemy drone manually. Note that the drone moves too quickly for manual targeting, and prompt the user to experience this difficulty. After they try and miss, the Code tab will be automatically unlocked.",
  codeUnlocked:
    "The Code tab has been unlocked. We won't be able to shoot the drone with manual aiming, its too fast. Clearly instruct the user to open the Code tab to develop an automated interception solution. We need a code that extract movement from a video comparing a frame to the next frame and making a difference between the two. This difference will highlight the movement of objects in the video. That data will be crutial for tracking the drone's position.",
  firstCode:
    "The user has entered the Code tab. As the Commander, task them with writing software to extract and track the drone's position from the camera feed (motion detection). Clearly specify that completion of this code is required before proceeding.",
  motionDetection:
    "The motion detection code has been completed and the 'Motion Detection' toggle has been automatically unlocked. Instruct the user to enable motion detection by toggling it on.",
  tracking:
    "The next step is critical. Instruct the user to create a tracking program that leverages the motion detection data to accurately follow the drone's position. Wait for the user to complete this tracking code.",
  trackingComplete:
    "The tracking code has been completed and the 'Tracking' toggle has been automatically unlocked. Guide the user to enable tracking, which allows continuous crosshair lock on the drone. Suggest that the user test the tracking system and attempt to fire.",
  positionPrediction:
    "Shots are missing due to drone speed. Explain to the user that they must implement position prediction using tracking data to shot where the drone will be instead of where it is now. Instruct the user to write code that predicts where the drone will be, and wait for them to finish this stage.",
  positionPredictionComplete:
    "The position prediction code has been completed and the 'Position Prediction' toggle has been automatically unlocked. Instruct the user to activate this feature, improving crosshair accuracy. Encourage another attempt at firing.",
  success:
    "The user has successfully intercepted the drone. Congratulate them formally. The Credits tab will be automatically unlocked when the drone is destroyed (health reaches 0).",
};

const initialCodeState: CodeState = {
  unlocked: false,
  typingSpeedIndex: 0, // Start at slowest speed
  codeProgress: 0, // Start at 0% progress
  codeTypedCode: "", // Start with empty code
};

export const useStore = create<StoreTypes>((set) => ({
  scenarioState: "intro",
  dataId: "0",
  unlockedCamera: true,
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
  activeLockTarget: false,
  droneHealth: 100, // Start at full health
  codeStates: {
    motionDetection: { ...initialCodeState },
    tracking: { ...initialCodeState },
    positionPrediction: { ...initialCodeState },
  },
  ttsQueue: [], // Start with empty TTS queue
  setScenarioState: (value) => set({ scenarioState: value }),
  setUnlockedCamera: (value) => set({ unlockedCamera: value }),
  setUnlockedMotion: (value) => set({ unlockedMotion: value }),
  setUnlockedFire: (value) => set({ unlockedFire: value }),
  setUnlockedTracking: (value) => set({ unlockedTracking: value }),
  setUnlockedAutoFire: (value) => set({ unlockedAutoFire: value }),
  setUnlockedMotionPrediction: (value) =>
    set({ unlockedMotionPrediction: value }),
  setUnlockedCode: (value) => {
    set((state) => {
      // Progress state: mission → codeUnlocked
      if (value && state.scenarioState === "mission") {
        highlightId("tab-motionDetection");
        return {
          unlockedCode: value,
          scenarioState: "codeUnlocked" as const,
          // Unlock motionDetection code when code tab is unlocked
          codeStates: {
            ...state.codeStates,
            motionDetection: {
              ...state.codeStates.motionDetection,
              unlocked: true,
            },
          },
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
  setDataId: (value) => set({ dataId: value }),
  setTab: (value) => {
    set((state) => {
      // codeUnlocked → firstCode: When motionDetection code tab is opened
      if (
        value === "motionDetection" &&
        state.scenarioState === "codeUnlocked" &&
        state.unlockedCode &&
        state.codeStates.motionDetection.unlocked
      ) {
        setTimeout(() => {
          useStore.getState().triggerAction("opened the code tab");
        }, 0);
        return {
          tab: value,
          scenarioState: "firstCode",
        };
      }
      // tracking → tracking: When tracking code tab is opened
      if (
        value === "tracking" &&
        state.scenarioState === "tracking" &&
        state.codeStates.tracking.unlocked
      ) {
        return { tab: value };
      }
      // positionPrediction → positionPrediction: When positionPrediction code tab is opened
      if (
        value === "positionPrediction" &&
        state.scenarioState === "positionPrediction" &&
        state.codeStates.positionPrediction.unlocked
      ) {
        return { tab: value };
      }
      return { tab: value };
    });
  },
  setActiveCamera: (value) => {
    set((state) => {
      // If camera is being turned on and we're in intro state, progress to mission and unlock fire
      if (value && state.scenarioState === "intro") {
        highlightId("left-click-to-fire");
        const newState = {
          activeCamera: value,
          unlockedFire: true,
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
        highlightId("tab-tracking");
        return {
          activeMotionDetection: value,
          activeCamera: true,
          scenarioState: "tracking",
          tab: "video",
          // Unlock tracking code when motion detection is enabled
          codeStates: {
            ...state.codeStates,
            tracking: {
              ...state.codeStates.tracking,
              unlocked: true,
            },
          },
        };
      }
      if (value) {
        return {
          activeMotionDetection: value,
          activeCamera: true,
          tab: "video",
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
          highlightId("tab-positionPrediction");
        }, 0);
        return {
          activeTracking: value,
          tab: "video",
          scenarioState: "positionPrediction",
          // Unlock positionPrediction code when tracking is enabled
          codeStates: {
            ...state.codeStates,
            positionPrediction: {
              ...state.codeStates.positionPrediction,
              unlocked: true,
            },
          },
        };
      }
      return { activeTracking: value };
    });
  },
  setActiveAutoFire: (value) => set({ activeAutoFire: value }),
  setActiveMotionPrediction: (value) => {
    if (value) {
      set({ activeMotionPrediction: value, tab: "video" });
    } else set({ activeMotionPrediction: value });
  },
  setActiveLockTarget: (value) =>
    set(() => {
      if (value) {
        // Also activate tracking when enabling lock target
        return {
          activeLockTarget: value,
          activeTracking: true,
        };
      }
      return { activeLockTarget: value };
    }),
  setDroneHealth: (value) =>
    set((state) => {
      const newHealth = Math.max(0, Math.min(100, value));
      // Win condition: when drone health reaches 0, set scenario state to success
      if (newHealth === 0 && state.scenarioState !== "success") {
        // Trigger action to notify commander
        setTimeout(() => {
          useStore
            .getState()
            .triggerAction("destroyed the enemy drone, victory achieved");
          highlightId("tab-credits");
        }, 0);
        return {
          droneHealth: newHealth,
          scenarioState: "success" as const,
          unlockedCredits: true,
        };
      }
      // Unlock credits when drone health reaches 0 (backward compatibility)
      if (newHealth === 0 && !state.unlockedCredits) {
        return {
          droneHealth: newHealth,
          unlockedCredits: true,
        };
      }
      return { droneHealth: newHealth };
    }),
  setCodeUnlocked: (codeKey, value) =>
    set((state) => ({
      codeStates: {
        ...state.codeStates,
        [codeKey]: {
          ...state.codeStates[codeKey],
          unlocked: value,
        },
      },
    })),
  setCodeTypingSpeedIndex: (codeKey, value) =>
    set((state) => ({
      codeStates: {
        ...state.codeStates,
        [codeKey]: {
          ...state.codeStates[codeKey],
          typingSpeedIndex: Number.isNaN(value)
            ? 0
            : Math.max(0, Math.min(3, value)),
        },
      },
    })),
  setCodeProgress: (codeKey, value) =>
    set((state) => ({
      codeStates: {
        ...state.codeStates,
        [codeKey]: {
          ...state.codeStates[codeKey],
          codeProgress: value,
        },
      },
    })),
  setCodeTypedCode: (codeKey, value) =>
    set((state) => ({
      codeStates: {
        ...state.codeStates,
        [codeKey]: {
          ...state.codeStates[codeKey],
          codeTypedCode: value,
        },
      },
    })),
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
  unlockAll: () => {
    set({
      unlockedCamera: true,
      unlockedMotion: true,
      unlockedFire: true,
      unlockedTracking: true,
      unlockedAutoFire: true,
      unlockedMotionPrediction: true,
      activeLockTarget: false,
      unlockedCode: true,
      unlockedCredits: true,
      scenarioState: "success",
      codeStates: {
        motionDetection: {
          ...initialCodeState,
          unlocked: true,
          codeProgress: 100,
        },
        tracking: { ...initialCodeState, unlocked: true, codeProgress: 100 },
        positionPrediction: {
          ...initialCodeState,
          unlocked: true,
          codeProgress: 100,
        },
      },
    });
  },
  reset: () =>
    set({
      scenarioState: "intro",
      tab: "video",
      activeCamera: true,
      activeMotionDetection: false,
      activeFire: false,
      activeTracking: false,
      activeAutoFire: false,
      activeMotionPrediction: false,
      activeLockTarget: false,
      droneHealth: 100,
      ttsQueue: [],
    }),
}));

// Export getState for use in callbacks
export const getStoreState = () => useStore.getState();
