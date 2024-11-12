const member_profile = document.querySelector(".member-profile");

const teamMembers = {
    member1: {
        image:"../static/photos/p1.jpg",
        name: "Robinson Karasha",
        github: "https://github.com/RobinsonKarasha?tab=repositories"
    },
    member2: {
        image:"../static/photos/p2.jpg",
        name: "Natalie Omondi",
        github: "https://github.com/akeyo03?tab=repositories"
    },
    member3: {
        image:"../static/photos/p3.jpg",
        name: "Martin Omondi",
        github: "https://github.com/ogutu-23?tab=repositories"
    },
    member4: {
        image:"../static/photos/p4.jpg",
        name: "Maureen Wambugu",
        github: "https://github.com/Mau-Wambugu?tab=repositories"
    },
    member5: {
        image:"../static/photos/p5.jpg",
        name: "James Kamau",
        github: "https://github.com/kamahTek?tab=repositories"
    },
    member6: {
        image:"../static/photos/p6.jpg",
        name: "Andrew Manwa",
        github: "https://github.com/sagwe0?tab=repositories"
    }
};

const icon = [
    '../static/icons/github.png',
];

for (let team in teamMembers){

    // gain access to our dictionary
    let member = teamMembers[team];

    // create a div that contains details about individual team member
    let individual_profile = document.createElement("div");

    // add class attribute to created <div> tags
    individual_profile.classList.add("card-profile", "m-0.5", "p-4");

    // add details inside the created <div>
    individual_profile.innerHTML = 
        `
        <img src="${member.image}" class="w-40 h-40 rounded-full"> <p>${member.name}</p> <a href=${member.github} target="_blank"><img src=${icon} class=" w-6 h-6 mt-2" /></a> 
        `;
    
    // append to the member_profile div
    member_profile.appendChild(individual_profile);

}