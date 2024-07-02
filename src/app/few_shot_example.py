
keypoints_exapmle = [
    {
        "question": """What are the keypoints of the meeting?:
        Alice: Good morning, everyone. Let's start with the project update. Bob, how is the development going?

Bob: Morning, Alice. The development is on track. We have completed the backend integration for the new feature. We are currently working on the frontend components.

Alice: Great. Carol, how about the design?

Carol: The design prototypes are ready. I've shared them with the team. We made a few changes based on the initial feedback.

Alice: Excellent. Dave, have you started the QA process?

Dave: Yes, we've started initial testing. So far, no major issues have been found, but we need to conduct more thorough tests once the frontend is complete.

Alice: That sounds good. Is there any risk or issue that we need to address immediately?

Bob: There's a potential delay in the frontend work if we encounter unexpected issues. We're doing our best to stay on schedule.

Carol: From the design side, everything looks fine. However, we need to ensure that the implementation matches the design closely to avoid rework.

Dave: No immediate risks from QA, but we need to stay vigilant as we move forward.

Alice: Thanks for the updates. Let's discuss the timeline. Bob, when do you think the frontend work will be complete?

Bob: We aim to complete it by next Wednesday. If there are any blockers, I'll inform the team immediately.

Alice: Good. Carol, once the frontend is done, how long will it take to verify the design alignment?

Carol: It should take a day or two to review everything.

Alice: And Dave, when can we expect the QA to be completed?

Dave: If the frontend is done by Wednesday and design verification by Friday, we should complete QA by the following Tuesday.

Alice: Perfect. Let's stick to this timeline and keep each other updated. Is there anything else anyone wants to discuss?

Carol: Just a quick note, I'll be on leave next Monday, but I'll ensure my tasks are completed beforehand.

Alice: Thanks for letting us know, Carol. If there's nothing else, we'll adjourn the meeting. Have a great day, everyone!
        """,
        "answer": """
        <kps> 1. Development update: Backend integration complete; frontend components in progress.</kps>
        <kps> 2. Design update: Prototypes ready and shared; minor changes based on feedback.</kps>
        <kps> 3. QA update: Initial testing started; no major issues so far; more thorough testing needed post-frontend completion.</kps>
        <kps> 4. Potential delay in frontend work if unexpected issues arise.</kps>
        <kps> 5. Importance of implementation matching the design to avoid rework.</kps>
        <kps> 6. Timeline:
            - Frontend completion: Next Wednesday
            - Design verification: 1-2 days post-frontend
            - QA completion: Following Tuesday</kps>
        <kps> 7. Carol on leave next Monday; will complete tasks beforehand.</kps>
        <kps> 8. Regular updates and adherence to the timeline emphasized.</kps>
        
        """
    }
]