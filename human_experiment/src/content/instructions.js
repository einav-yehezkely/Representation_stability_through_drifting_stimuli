// RESEARCHER-EDITABLE participant instructions (PRD §4). Independent of scientific logic.
// Must not mention that the categories change during the experiment.

export const instructionsContent = {
  title: "Instructions",
  paragraphs: [
    "In this study you will see pictures of faces, one at a time.",
    "Each face belongs to one of two groups: Group A or Group B. Your task is to learn which faces belong to which group.",
    "For each face, choose one of four answers:",
  ],
  responseExplanations: [
    { label: "A", text: "you are confident the face belongs to Group A" },
    { label: "Probably A", text: "you think the face belongs to Group A, but you are not sure" },
    { label: "Probably B", text: "you think the face belongs to Group B, but you are not sure" },
    { label: "B", text: "you are confident the face belongs to Group B" },
  ],
  closingParagraphs: [
    "At the beginning, you will be told after each answer whether it was correct or incorrect. Use this feedback to learn the two groups.",
    "At first you will have to guess. That is expected.",
  ],
  startLabel: "Start",
};
