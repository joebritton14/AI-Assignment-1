import random
import numpy as np
import matplotlib.pyplot as plt
import csv

student_list = []
supervisor_list = []
with open('Student-choices.csv', 'r') as stud_file:
    reader = csv.reader(stud_file)
    students = list(reader)

with open("Supervisors.csv", "r") as sup_file:
    reader = csv.reader(sup_file)
    supervisors = list(reader)

for student in students:
    student_list.append([student[0], student[1:]])


for supervisor in supervisors:
    supervisor_list.append([supervisor[0], int(supervisor[1]), 0])

print(student_list)
print(supervisor_list)
pop_size = 1000


def generate_initial_population(student_list, supervisor_list, population_size):
    population = []
    for i in range(population_size):
        mapping = {}
        for student in student_list:
            # choose random lecturer
            lecturer = random.choice(supervisor_list)
            # if lecturer is at capacity, change to different lecturer
            while lecturer[2] >= lecturer[1]:
                lecturer = random.choice(supervisor_list)

            # map the student to the lecturer
            mapping[student[0]] = lecturer[0]

            # update lecturer student count
            lecturer[2] += 1
        population.append(mapping)

        # reset the lecturer student count
        for lect in supervisor_list:
            lect[2] = 0

    return population


def mutate(pop):
    # get 2 random students and swap one of their supervisors
    # ensure these are not the same lecturer
    # 1/3 chance of mutation
    chance = random.randint(0, 8)
    if chance < 3:

        pos1 = random.randint(0, len(student_list)-1)
        pos2 = random.randint(0, len(student_list)-1)
        while pos1 == pos2:
            pos2 = random.randint(0, len(student_list)-1)

        stud1, stud2 = student_list[pos1], student_list[pos2]

        # swapping the lecturers of student 1 and 2
        # store sup in temp
        temp = pop[stud1[0]]
        pop[stud1[0]] = pop[stud2[0]]
        pop[stud2[0]] = temp

    return pop


def fitness(pop):
    # optimal fitness = 22, worst fitness = 1012
    fitness_score = 22
    worst_fitness = 22
    student_choice_total = 0

    for stud in pop:
        supervisor_index = int(pop[stud][11:]) - 1
        student_index = int(stud[8:]) - 1

        ranking = int(student_list[student_index][1][supervisor_index])
        student_choice_total += ranking
        # fitness_score += ranking
        # fitness_score += stud.rankings[stud.supervisor.ID - 1]
    # print("FITNESS = ", fitness_score, " - ", student_choice_total, " = ", (fitness_score - student_choice_total))

    student_choice_total = round(student_choice_total / 46, 4)
    fitness_score -= student_choice_total
    print("STUD AVG = ", student_choice_total)
    print(fitness_score)
    print("Average choice = ", round(student_choice_total))
    return fitness_score


def one_point_crossover(p1, p2):
    point = random.randint(1, len(p1) - 1)

    child1 = {}
    child2 = {}
    child1.update(dict(list(p1.items())[:point]))
    child1.update(dict(list(p2.items())[point:]))
    child2.update(dict(list(p2.items())[:point]))
    child2.update(dict(list(p1.items())[point:]))


    # check capacity of lecturers, assign to new lect if necessary
    for stud in child1:
        # find the index of the supervisor associated with this student
        supervisor_index = int(child1[stud][11:]) - 1
        # grab lecturer using index
        lecturer = supervisor_list[supervisor_index]

        # if a student is with a lecturer that is at capacity, change lect
        while lecturer[2] >= lecturer[1]:
            lecturer = random.choice(supervisor_list)

        child1[stud] = lecturer[0]
        lecturer[2] += 1

    # reset lecturer student count
    for lect in supervisor_list:
        lect[2] = 0

    # start from the back on this one
    i = len(child2) - 1

    while i >= 0:
        supervisor_index = int(child2[stud][11:]) - 1
        lecturer = supervisor_list[supervisor_index]

        while lecturer[2] >= lecturer[1]:
            lecturer = random.choice(supervisor_list)

        child2[stud] = lecturer[0]
        lecturer[2] += 1

        i -= 1

    for lect in supervisor_list:
        lect[2] = 0

    return child1, child2


def perform(fitness_func, population, mutation):
    generations = 100
    avg_fit = []
    for _ in range(generations):
        # Evaluate fitness
        fit = [fitness_func(ind) for ind in population]

        # Select parents using biased roulette wheel selection
        if sum(fit) > 0:
            probability = np.true_divide(fit, sum(fit))
            parents = np.random.choice(population, size=pop_size, replace=True, p=probability)
        else:
            parents = population

        # Generate offspring
        offspring = []
        for i in range(0, pop_size - 1, 2):
            ind1, ind2 = parents[i], parents[i + 1]
            ind1, ind2 = one_point_crossover(ind1, ind2)
            ind1, ind2 = mutation(ind1), mutation(ind2)
            offspring.extend([ind1, ind2])

        # Update population
        population = offspring

        # Calculate average fitness
        avg_fit.append(sum(fit) / len(fit))

    # Plot average fitness
    plt.plot(avg_fit)
    plt.xlabel('Generations')
    plt.ylabel('Average Fitness Using ' + fitness_func.__name__)
    plt.show()


populations = generate_initial_population(student_list, supervisor_list, pop_size)
print(populations)
perform(fitness, populations, mutate)
