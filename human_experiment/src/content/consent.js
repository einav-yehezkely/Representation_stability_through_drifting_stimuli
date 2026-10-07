// RESEARCHER-EDITABLE study information shown on the consent screen.
//
// Every [RESEARCHER: ...] marker is a placeholder that must be replaced with finalized,
// approved text before data collection. Nothing here is approved consent language; no approval
// numbers or institutional claims have been invented.

export const consentContent = {
  title: "[RESEARCHER: Study title]",
  sections: [
    {
      heading: "About this study",
      body: "[RESEARCHER: Short description of the study purpose, written for participants.]",
    },
    {
      heading: "What you will do",
      body:
        "[RESEARCHER: Description of the task. For example: you will see pictures of faces, one at a time, and decide which group each face belongs to.] " +
        "[RESEARCHER: Expected duration.]",
    },
    {
      heading: "Risks and benefits",
      body: "[RESEARCHER: Risks, discomforts and benefits.]",
    },
    {
      heading: "Compensation",
      body: "[RESEARCHER: Payment / compensation details.]",
    },
    {
      heading: "Your data",
      body: "[RESEARCHER: What data are recorded, how they are stored, who can access them, and how long they are kept.]",
    },
    {
      heading: "Voluntary participation",
      body: "[RESEARCHER: Statement that participation is voluntary and that participants may stop at any time, and what happens if they do.]",
    },
    {
      heading: "Ethics approval and contact",
      body: "[RESEARCHER: Ethics committee / IRB name and approval number.] [RESEARCHER: Researcher contact details.]",
    },
  ],
  // Exact wording required by the PRD (§4).
  checkboxLabel: "I have read the information above and agree to participate in this study.",
  continueLabel: "Continue",
};
